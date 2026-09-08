//! PostgreSQL source of truth. All shelf relations and extension objects are qualified.
//! Versions use a non-cycling sequence, so delete/recreate cannot revive stale jobs.
//!
//! TLS uses native-tls defaults: platform trust roots and certificate/hostname
//! verification are enabled. The driver accepts sslmode=disable/prefer/require;
//! prefer is its default and permits plaintext when the server declines TLS.
//! Use require to mandate TLS. libpq modes verify-ca/verify-full and sslrootcert
//! connection options are unsupported and fail configuration parsing; they are
//! not silently ignored. Custom roots/client identities are not configured here.
use crate::{
    embedding::EmbeddingConfig,
    error::{HypatiaError, Result},
    model::{Content, Knowledge, SearchOpts, Statement, StatementKey},
    storage::{
        settings::{PostgresSettings, VectorSettings},
        transfer::{
            EmbeddingMetadata, KnowledgeRecord, Snapshot, StatementRecord, validate_vector,
        },
    },
};
use chrono::NaiveDateTime;
use pgvector::Vector;
use postgres::{
    Client, GenericClient, Row,
    types::{ToSql, Type},
};
use serde_json::Value;
use std::{cell::RefCell, time::Duration};

use super::FtsResult;
pub struct PgStore {
    client: RefCell<Client>,
    schema: String,
    extension: String,
    dimensions: usize,
    model: String,
}
// Do not echo connection strings, SQL text, or server DETAIL (which can carry data).
fn pg_error(e: postgres::Error) -> HypatiaError {
    let code = e.code().map(|c| c.code()).unwrap_or("connection/protocol");
    let reason = match code {
        "23505" => "duplicate key",
        "23502" | "23503" | "23514" => "data constraint violation",
        "42501" => "insufficient database privileges",
        "57014" => "statement cancelled or timeout exceeded",
        "40P01" => "transaction deadlock; retry the operation",
        "40001" => "serialization conflict; retry the operation",
        "42P01" | "42703" | "42883" => "required database object or function is missing",
        "28P01" | "28000" => "authentication failed",
        "connection/protocol" => "connection or parameter encoding failure",
        _ => "database operation failed",
    };
    let message = format!("PostgreSQL: {reason} ({code})");
    if code.starts_with("23") {
        HypatiaError::Validation(message)
    } else {
        HypatiaError::Config(message)
    }
}
fn invalid(s: impl Into<String>) -> HypatiaError {
    HypatiaError::Validation(s.into())
}
fn ident(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}
fn catalog_info(catalog: &str) -> Result<(&'static str, &'static str)> {
    match catalog {
        "knowledge" => Ok(("knowledge", "name")),
        "statement" => Ok(("statement", "triple")),
        _ => Err(invalid("catalog must be knowledge or statement")),
    }
}
fn validate_timestamp(value: NaiveDateTime) -> Result<()> {
    use chrono::Timelike;
    if value.nanosecond() % 1000 != 0 || value.nanosecond() >= 1_000_000_000 {
        return Err(invalid(
            "PostgreSQL timestamps require microsecond precision and do not preserve leap-second encodings",
        ));
    }
    Ok(())
}

fn content_values(c: &Content) -> Result<(Value, Option<Value>, Value)> {
    Ok((
        serde_json::to_value(c)?,
        serde_json::from_str(&c.data).ok(),
        crate::engine::postgres::membership_tokens(&c.to_json_string()),
    ))
}
fn knowledge_row(r: &Row) -> Result<Knowledge> {
    Ok(Knowledge {
        name: r.try_get("name").map_err(pg_error)?,
        content: serde_json::from_value(r.try_get::<_, Value>("content").map_err(pg_error)?)?,
        created_at: r.try_get("created_at").map_err(pg_error)?,
    })
}
fn statement_row(r: &Row) -> Result<Statement> {
    Ok(Statement {
        key: StatementKey {
            head: r.try_get("head").map_err(pg_error)?,
            relation: r.try_get("relation").map_err(pg_error)?,
            tail: r.try_get("tail").map_err(pg_error)?,
        },
        content: serde_json::from_value(r.try_get::<_, Value>("content").map_err(pg_error)?)?,
        created_at: r.try_get("created_at").map_err(pg_error)?,
        tr_start: r.try_get("tr_start").map_err(pg_error)?,
        tr_end: r.try_get("tr_end").map_err(pg_error)?,
    })
}
impl PgStore {
    pub fn open(
        config: &PostgresSettings,
        vector: &VectorSettings,
        embedding: &EmbeddingConfig,
    ) -> Result<Self> {
        super::settings::validate_schema(&config.schema)?;
        if !embedding.model_identity_trusted {
            return Err(HypatiaError::Config("PostgreSQL requires a trusted embedding model identity; configure embedding.model or readable model and tokenizer files".into()));
        }
        if vector.metric != "cosine" || !matches!(vector.index.as_str(), "hnsw" | "none") {
            return Err(HypatiaError::Config(
                "PostgreSQL vectors require metric=cosine and index=hnsw or none".into(),
            ));
        }
        let dimensions = embedding.dimensions();
        if dimensions == 0 || dimensions > 16000 || (vector.index == "hnsw" && dimensions > 2000) {
            return Err(HypatiaError::Config(
                "pgvector dimensions must be 1–16000 (HNSW maximum: 2000)".into(),
            ));
        }
        if config.connect_timeout_seconds == 0
            || config.statement_timeout_ms == 0
            || config.statement_timeout_ms > i32::MAX as u64
        {
            return Err(HypatiaError::Config(
                "PostgreSQL timeouts must be positive; statement timeout must fit i32 milliseconds"
                    .into(),
            ));
        }
        let url = config.connection_url()?;
        let mut connection: postgres::Config = url.parse().map_err(|_| {
            HypatiaError::Config("Invalid PostgreSQL connection configuration".into())
        })?;
        connection.connect_timeout(Duration::from_secs(config.connect_timeout_seconds));
        let tls = native_tls::TlsConnector::builder()
            .build()
            .map_err(|_| HypatiaError::Config("Cannot initialize PostgreSQL TLS".into()))?;
        let mut client = connection
            .connect(postgres_native_tls::MakeTlsConnector::new(tls))
            .map_err(pg_error)?;
        client
            .query_one(
                "SELECT pg_catalog.set_config('statement_timeout',$1,false)",
                &[&config.statement_timeout_ms.to_string()],
            )
            .map_err(pg_error)?;
        client
            .batch_execute("SET search_path = pg_catalog")
            .map_err(pg_error)?;
        let extension:String=client.query_opt("SELECT n.nspname FROM pg_catalog.pg_extension e JOIN pg_catalog.pg_namespace n ON n.oid=e.extnamespace WHERE e.extname='vector'",&[]).map_err(pg_error)?.ok_or_else(||HypatiaError::Config("The vector extension must be preinstalled by a database administrator".into()))?.get(0);
        let store = Self {
            client: RefCell::new(client),
            schema: config.schema.clone(),
            extension: ident(&extension),
            dimensions,
            model: embedding.model_identity().to_string(),
        };
        store.init_schema(&vector.index)?;
        Ok(store)
    }
    pub fn schema(&self) -> &str {
        &self.schema
    }
    fn table(&self, t: &str) -> String {
        format!("{}.{}", ident(&self.schema), ident(t))
    }
    fn init_schema(&self, index: &str) -> Result<()> {
        let mut client = self.client.borrow_mut();
        let mut tx = client.transaction().map_err(pg_error)?;
        tx.query_one(
            "SELECT pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended($1,0))",
            &[&format!("hypatia:{}", self.schema)],
        )
        .map_err(pg_error)?;
        tx.batch_execute(&format!("CREATE SCHEMA IF NOT EXISTS {}; CREATE TABLE IF NOT EXISTS {}(k text PRIMARY KEY,v text NOT NULL)",ident(&self.schema),self.table("meta"))).map_err(pg_error)?;
        let expected = [
            ("schema_version", "1".to_string()),
            ("embedding_model", self.model.clone()),
            ("embedding_dimensions", self.dimensions.to_string()),
            ("vector_metric", "cosine".to_string()),
            ("vector_index", index.to_string()),
        ];
        let metadata = tx
            .query(&format!("SELECT k,v FROM {}", self.table("meta")), &[])
            .map_err(pg_error)?;
        if !metadata.is_empty() {
            for (key, value) in &expected {
                if !metadata
                    .iter()
                    .any(|r| r.get::<_, String>(0) == *key && r.get::<_, String>(1) == *value)
                {
                    return Err(HypatiaError::Config(format!(
                        "PostgreSQL shelf metadata mismatch: {key}; explicit migration or re-embedding is required"
                    )));
                }
            }
            for table in ["knowledge", "statement", "docs", "content_versions"] {
                let exists: bool = tx
                    .query_one(
                        "SELECT pg_catalog.to_regclass($1) IS NOT NULL",
                        &[&self.table(table)],
                    )
                    .map_err(pg_error)?
                    .get(0);
                if !exists {
                    return Err(HypatiaError::Config(format!(
                        "PostgreSQL shelf object missing: {table}"
                    )));
                }
            }
            self.validate_schema_structure(&mut tx, index)?;
            return tx.commit().map_err(pg_error);
        }
        tx.batch_execute(&format!(r#"
CREATE SEQUENCE {versions} AS bigint NO CYCLE;
CREATE TABLE {docs}(id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,catalog text NOT NULL CHECK(catalog IN ('knowledge','statement')),key text NOT NULL,search_vector tsvector NOT NULL,UNIQUE(catalog,key));
CREATE INDEX docs_search_idx ON {docs} USING gin(search_vector);
CREATE TABLE {knowledge}(name text PRIMARY KEY,content jsonb NOT NULL,payload jsonb,tokens jsonb NOT NULL,created_at timestamp NOT NULL DEFAULT(CURRENT_TIMESTAMP AT TIME ZONE 'UTC'),content_version bigint NOT NULL DEFAULT nextval('{versions}'::regclass),embedding {ext}.vector({dims}));
CREATE TABLE {statement}(triple text PRIMARY KEY,head text NOT NULL,relation text NOT NULL,tail text NOT NULL,content jsonb NOT NULL,payload jsonb,tokens jsonb NOT NULL,created_at timestamp NOT NULL DEFAULT(CURRENT_TIMESTAMP AT TIME ZONE 'UTC'),tr_start timestamp,tr_end timestamp,content_version bigint NOT NULL DEFAULT nextval('{versions}'::regclass),embedding {ext}.vector({dims}));
CREATE INDEX knowledge_payload_idx ON {knowledge} USING gin(payload jsonb_path_ops);
CREATE INDEX statement_payload_idx ON {statement} USING gin(payload jsonb_path_ops);
CREATE INDEX statement_head_idx ON {statement}(head);
CREATE INDEX statement_relation_idx ON {statement}(relation);
CREATE INDEX statement_tail_idx ON {statement}(tail);
CREATE INDEX knowledge_missing_embedding_idx ON {knowledge}(name) WHERE embedding IS NULL;
CREATE INDEX statement_missing_embedding_idx ON {statement}(triple) WHERE embedding IS NULL;
"#,versions=self.table("content_versions"),docs=self.table("docs"),knowledge=self.table("knowledge"),statement=self.table("statement"),ext=self.extension,dims=self.dimensions)).map_err(pg_error)?;
        tx.batch_execute(&crate::engine::postgres::schema_functions(&self.schema))
            .map_err(pg_error)?;
        if index == "hnsw" {
            for table in ["knowledge", "statement"] {
                tx.batch_execute(&format!(
                    "CREATE INDEX {} ON {} USING hnsw(embedding {}.vector_cosine_ops)",
                    ident(&format!("{table}_embedding_hnsw_idx")),
                    self.table(table),
                    self.extension
                ))
                .map_err(pg_error)?;
            }
        }
        for (key, value) in expected {
            tx.execute(
                &format!("INSERT INTO {}(k,v) VALUES($1,$2)", self.table("meta")),
                &[&key, &value],
            )
            .map_err(pg_error)?;
        }
        tx.commit().map_err(pg_error)
    }
    fn validate_schema_structure(&self, tx: &mut impl GenericClient, index: &str) -> Result<()> {
        for table in ["knowledge", "statement"] {
            let rows = tx.query("SELECT a.attname,t.typname,n.nspname,a.atttypmod,a.attnotnull FROM pg_catalog.pg_attribute a JOIN pg_catalog.pg_type t ON t.oid=a.atttypid JOIN pg_catalog.pg_namespace n ON n.oid=t.typnamespace WHERE a.attrelid=pg_catalog.to_regclass($1) AND a.attnum>0 AND NOT a.attisdropped", &[&self.table(table)]).map_err(pg_error)?;
            for (column, typ, not_null) in [
                ("content", "jsonb", true),
                ("payload", "jsonb", false),
                ("tokens", "jsonb", true),
                ("content_version", "int8", true),
                ("embedding", "vector", false),
            ] {
                let valid = rows.iter().any(|r| {
                    r.get::<_, String>(0) == column
                        && r.get::<_, String>(1) == typ
                        && r.get::<_, bool>(4) == not_null
                        && if column == "embedding" {
                            ident(&r.get::<_, String>(2)) == self.extension
                                && r.get::<_, i32>(3) == self.dimensions as i32
                        } else {
                            r.get::<_, String>(2) == "pg_catalog"
                        }
                });
                if !valid {
                    return Err(HypatiaError::Config(format!(
                        "PostgreSQL schema mismatch: {table}.{column}"
                    )));
                }
            }
            if index == "hnsw" {
                let valid: bool = tx.query_one("SELECT EXISTS(SELECT 1 FROM pg_catalog.pg_index i JOIN pg_catalog.pg_class c ON c.oid=i.indexrelid JOIN pg_catalog.pg_am a ON a.oid=c.relam JOIN pg_catalog.pg_opclass o ON o.oid=i.indclass[0] WHERE i.indexrelid=pg_catalog.to_regclass($1) AND i.indrelid=pg_catalog.to_regclass($2) AND i.indisvalid AND a.amname='hnsw' AND o.opcname='vector_cosine_ops')", &[&self.table(&format!("{table}_embedding_hnsw_idx")), &self.table(table)]).map_err(pg_error)?.get(0);
                if !valid {
                    return Err(HypatiaError::Config(format!(
                        "PostgreSQL HNSW index is missing or invalid: {table}"
                    )));
                }
            }
        }
        for signature in [
            "jse_numeric(text)",
            "jse_json_contains(jsonb,jsonb)",
            "jse_compact(jsonb)",
        ] {
            let qualified = format!("{}.{}", ident(&self.schema), signature);
            let exists: bool = tx
                .query_one(
                    "SELECT pg_catalog.to_regprocedure($1) IS NOT NULL",
                    &[&qualified],
                )
                .map_err(pg_error)?
                .get(0);
            if !exists {
                return Err(HypatiaError::Config(format!(
                    "PostgreSQL helper function missing: {signature}"
                )));
            }
        }
        Ok(())
    }
    fn upsert_doc(
        &self,
        tx: &mut impl GenericClient,
        catalog: &str,
        key: &str,
        content: &Content,
    ) -> Result<()> {
        let f = content.fts_fields(key);
        tx.execute(&format!("INSERT INTO {}(catalog,key,search_vector) VALUES($1,$2,setweight(to_tsvector('pg_catalog.simple'::regconfig,$3),'A') || setweight(to_tsvector('pg_catalog.simple'::regconfig,$4),'D') || setweight(to_tsvector('pg_catalog.simple'::regconfig,$5),'B') || setweight(to_tsvector('pg_catalog.simple'::regconfig,$6),'C')) ON CONFLICT(catalog,key) DO UPDATE SET search_vector=excluded.search_vector",self.table("docs")),&[&catalog,&key,&f.key,&f.data,&f.tags,&f.synonyms]).map_err(pg_error)?;
        Ok(())
    }
    pub fn insert_knowledge(&self, name: &str, content: &Content) -> Result<i64> {
        let (raw, payload, tokens) = content_values(content)?;
        let mut client = self.client.borrow_mut();
        let mut tx = client.transaction().map_err(pg_error)?;
        let token=tx.query_one(&format!("INSERT INTO {}(name,content,payload,tokens) VALUES($1,$2,$3,$4) RETURNING content_version",self.table("knowledge")),&[&name,&raw,&payload,&tokens]).map_err(pg_error)?.get(0);
        self.upsert_doc(&mut tx, "knowledge", name, content)?;
        tx.commit().map_err(pg_error)?;
        Ok(token)
    }
    pub fn update_knowledge(&self, name: &str, content: &Content) -> Result<i64> {
        let (raw, payload, tokens) = content_values(content)?;
        let mut client = self.client.borrow_mut();
        let mut tx = client.transaction().map_err(pg_error)?;
        let token=tx.query_opt(&format!("UPDATE {} SET content=$2,payload=$3,tokens=$4,embedding=NULL,content_version=nextval('{}'::regclass) WHERE name=$1 RETURNING content_version",self.table("knowledge"),self.table("content_versions")),&[&name,&raw,&payload,&tokens]).map_err(pg_error)?.ok_or_else(||HypatiaError::NotFound{kind:"knowledge".into(),key:name.into()})?.get(0);
        self.upsert_doc(&mut tx, "knowledge", name, content)?;
        tx.commit().map_err(pg_error)?;
        Ok(token)
    }
    pub fn get_knowledge(&self, name: &str) -> Result<Option<Knowledge>> {
        self.client
            .borrow_mut()
            .query_opt(
                &format!(
                    "SELECT name,content,created_at FROM {} WHERE name=$1",
                    self.table("knowledge")
                ),
                &[&name],
            )
            .map_err(pg_error)?
            .as_ref()
            .map(knowledge_row)
            .transpose()
    }
    pub fn insert_statement(
        &self,
        key: &StatementKey,
        content: &Content,
        tr_start: Option<NaiveDateTime>,
        tr_end: Option<NaiveDateTime>,
    ) -> Result<i64> {
        for value in [tr_start, tr_end].into_iter().flatten() {
            validate_timestamp(value)?;
        }
        let (raw, payload, tokens) = content_values(content)?;
        let triple = key.to_csv_key();
        let mut client = self.client.borrow_mut();
        let mut tx = client.transaction().map_err(pg_error)?;
        let token=tx.query_one(&format!("INSERT INTO {}(triple,head,relation,tail,content,payload,tr_start,tr_end,tokens) VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9) RETURNING content_version",self.table("statement")),&[&triple,&key.head,&key.relation,&key.tail,&raw,&payload,&tr_start,&tr_end,&tokens]).map_err(pg_error)?.get(0);
        self.upsert_doc(&mut tx, "statement", &triple, content)?;
        tx.commit().map_err(pg_error)?;
        Ok(token)
    }
    pub fn update_statement(
        &self,
        key: &StatementKey,
        content: &Content,
        tr_start: Option<NaiveDateTime>,
        tr_end: Option<NaiveDateTime>,
    ) -> Result<i64> {
        for value in [tr_start, tr_end].into_iter().flatten() {
            validate_timestamp(value)?;
        }
        let (raw, payload, tokens) = content_values(content)?;
        let triple = key.to_csv_key();
        let mut client = self.client.borrow_mut();
        let mut tx = client.transaction().map_err(pg_error)?;
        let token=tx.query_opt(&format!("UPDATE {} SET content=$2,payload=$3,tr_start=$4,tr_end=$5,tokens=$6,embedding=NULL,content_version=nextval('{}'::regclass) WHERE triple=$1 RETURNING content_version",self.table("statement"),self.table("content_versions")),&[&triple,&raw,&payload,&tr_start,&tr_end,&tokens]).map_err(pg_error)?.ok_or_else(||HypatiaError::NotFound{kind:"statement".into(),key:triple.clone()})?.get(0);
        self.upsert_doc(&mut tx, "statement", &triple, content)?;
        tx.commit().map_err(pg_error)?;
        Ok(token)
    }
    pub fn get_statement(&self, key: &StatementKey) -> Result<Option<Statement>> {
        self.client
            .borrow_mut()
            .query_opt(
                &format!("SELECT * FROM {} WHERE triple=$1", self.table("statement")),
                &[&key.to_csv_key()],
            )
            .map_err(pg_error)?
            .as_ref()
            .map(statement_row)
            .transpose()
    }
    fn delete(&self, catalog: &str, key: &str) -> Result<()> {
        let (table, pk) = catalog_info(catalog)?;
        let mut client = self.client.borrow_mut();
        let mut tx = client.transaction().map_err(pg_error)?;
        if tx
            .execute(
                &format!("DELETE FROM {} WHERE {pk}=$1", self.table(table)),
                &[&key],
            )
            .map_err(pg_error)?
            == 0
        {
            return Err(HypatiaError::NotFound {
                kind: catalog.into(),
                key: key.into(),
            });
        }
        tx.execute(
            &format!(
                "DELETE FROM {} WHERE catalog=$1 AND key=$2",
                self.table("docs")
            ),
            &[&catalog, &key],
        )
        .map_err(pg_error)?;
        tx.commit().map_err(pg_error)
    }
    pub fn delete_knowledge(&self, name: &str) -> Result<()> {
        self.delete("knowledge", name)
    }
    pub fn delete_statement(&self, key: &StatementKey) -> Result<()> {
        self.delete("statement", &key.to_csv_key())
    }
    fn query(&self, sql: &str, values: Vec<Value>) -> Result<Vec<Row>> {
        let mut client = self.client.borrow_mut();
        let stmt = client.prepare(sql).map_err(pg_error)?;
        if values.len() != stmt.params().len() {
            return Err(invalid("query parameter count mismatch"));
        }
        let params: Vec<Box<dyn ToSql + Sync>> = values
            .iter()
            .zip(stmt.params())
            .map(|(v, t)| parameter(v, t))
            .collect::<Result<_>>()?;
        let refs: Vec<&(dyn ToSql + Sync)> = params.iter().map(|p| p.as_ref()).collect();
        client.query(&stmt, &refs).map_err(pg_error)
    }
    pub fn query_knowledge(&self, sql: &str, params: Vec<Value>) -> Result<Vec<Knowledge>> {
        self.query(sql, params)?.iter().map(knowledge_row).collect()
    }
    pub fn query_statements(&self, sql: &str, params: Vec<Value>) -> Result<Vec<Statement>> {
        self.query(sql, params)?.iter().map(statement_row).collect()
    }
    pub fn query_khop(
        &self,
        head: &str,
        relation: Option<&str>,
        depth: i64,
    ) -> Result<Vec<Statement>> {
        if depth <= 0 {
            return Ok(vec![]);
        }
        if depth > 100 {
            return Err(invalid("k-hop depth cannot exceed 100"));
        }
        // UNION deduplicates identical (edge,depth) states, bounding cyclic expansion.
        let sql = format!(
            "WITH RECURSIVE hop AS (SELECT triple,tail,1::bigint AS depth FROM {s} WHERE head=$1 AND ($2::text IS NULL OR relation=$2) UNION SELECT s.triple,s.tail,h.depth+1 FROM hop h JOIN {s} s ON s.head=h.tail WHERE h.depth<$3 AND ($2::text IS NULL OR s.relation=$2)), nearest AS (SELECT triple,min(depth) AS depth FROM hop GROUP BY triple) SELECT s.* FROM nearest n JOIN {s} s ON s.triple=n.triple ORDER BY n.depth,s.created_at DESC,s.triple",
            s = self.table("statement")
        );
        self.client
            .borrow_mut()
            .query(&sql, &[&head, &relation, &depth])
            .map_err(pg_error)?
            .iter()
            .map(statement_row)
            .collect()
    }
    pub fn embedding_version(&self, catalog: &str, key: &str) -> Result<Option<i64>> {
        let (table, pk) = catalog_info(catalog)?;
        Ok(self
            .client
            .borrow_mut()
            .query_opt(
                &format!(
                    "SELECT content_version FROM {} WHERE {pk}=$1",
                    self.table(table)
                ),
                &[&key],
            )
            .map_err(pg_error)?
            .map(|r| r.get(0)))
    }
    /// Single conditional UPDATE is the linearization point for background work.
    pub fn install_embedding(
        &self,
        catalog: &str,
        key: &str,
        version: i64,
        vector: &[f32],
    ) -> Result<bool> {
        let (table, pk) = catalog_info(catalog)?;
        validate_vector(vector, self.dimensions)?;
        let vector = Vector::from(vector.to_vec());
        Ok(self
            .client
            .borrow_mut()
            .execute(
                &format!(
                    "UPDATE {} SET embedding=$3 WHERE {pk}=$1 AND content_version=$2",
                    self.table(table)
                ),
                &[&key, &version, &vector],
            )
            .map_err(pg_error)?
            == 1)
    }
    /// Explicit re-embedding reset. Every row receives a new token, including
    /// already-missing embeddings, so all outstanding workers are invalidated.
    pub fn clear_all_embeddings(&self) -> Result<()> {
        let mut client = self.client.borrow_mut();
        let mut tx = client.transaction().map_err(pg_error)?;
        for table in ["knowledge", "statement"] {
            tx.execute(
                &format!(
                    "UPDATE {} SET embedding=NULL,content_version=nextval('{}'::regclass)",
                    self.table(table),
                    self.table("content_versions")
                ),
                &[],
            )
            .map_err(pg_error)?;
        }
        tx.commit().map_err(pg_error)
    }
    fn clear_embedding(&self, catalog: &str, key: &str) -> Result<()> {
        let (table, pk) = catalog_info(catalog)?;
        self.client.borrow_mut().execute(&format!("UPDATE {} SET embedding=NULL,content_version=nextval('{}'::regclass) WHERE {pk}=$1",self.table(table),self.table("content_versions")),&[&key]).map_err(pg_error)?;
        Ok(())
    }
    pub fn clear_knowledge_embedding(&self, name: &str) -> Result<()> {
        self.clear_embedding("knowledge", name)
    }
    pub fn clear_statement_embedding(&self, triple: &str) -> Result<()> {
        self.clear_embedding("statement", triple)
    }
    pub fn missing_embeddings(
        &self,
        catalog: &str,
        after_key: Option<&str>,
        limit: i64,
    ) -> Result<Vec<(String, Content, i64)>> {
        let (table, pk) = catalog_info(catalog)?;
        if limit < 0 {
            return Err(invalid("embedding page limit must be nonnegative"));
        }
        self.client.borrow_mut().query(&format!("SELECT {pk},content,content_version FROM {} WHERE embedding IS NULL AND ($1::text IS NULL OR {pk}>$1) ORDER BY {pk} LIMIT $2",self.table(table)),&[&after_key,&limit]).map_err(pg_error)?.iter().map(|r|Ok((r.get(0),serde_json::from_value(r.get(1))?,r.get(2)))).collect()
    }
    fn entries(&self, catalog: &str, present: bool) -> Result<Vec<(String, String)>> {
        let (table, pk) = catalog_info(catalog)?;
        Ok(self.client.borrow_mut().query(&format!("SELECT {pk},content::text FROM {} WHERE (embedding IS NOT NULL)=$1 ORDER BY {pk}",self.table(table)),&[&present]).map_err(pg_error)?.iter().map(|r|(r.get(0),r.get(1))).collect())
    }
    pub fn knowledge_with_embeddings(&self) -> Result<Vec<(String, String)>> {
        self.entries("knowledge", true)
    }
    pub fn knowledge_without_embeddings(&self) -> Result<Vec<(String, String)>> {
        self.entries("knowledge", false)
    }
    pub fn statements_without_embeddings(&self) -> Result<Vec<(String, String)>> {
        self.entries("statement", false)
    }
    pub fn embedding_row_count(&self, catalog: &str) -> Result<usize> {
        let (table, _) = catalog_info(catalog)?;
        let n: i64 = self
            .client
            .borrow_mut()
            .query_one(
                &format!(
                    "SELECT count(*) FROM {} WHERE embedding IS NOT NULL",
                    self.table(table)
                ),
                &[],
            )
            .map_err(pg_error)?
            .get(0);
        Ok(n as usize)
    }
    pub fn doc_id_by_key(&self, catalog: &str, key: &str) -> Result<Option<i64>> {
        catalog_info(catalog)?;
        Ok(self
            .client
            .borrow_mut()
            .query_opt(
                &format!(
                    "SELECT id FROM {} WHERE catalog=$1 AND key=$2",
                    self.table("docs")
                ),
                &[&catalog, &key],
            )
            .map_err(pg_error)?
            .map(|r| r.get(0)))
    }
    pub fn rows_by_doc_ids(
        &self,
        catalog: &str,
        ids: &[i64],
    ) -> Result<Vec<(i64, String, String)>> {
        let (table, pk) = catalog_info(catalog)?;
        Ok(self.client.borrow_mut().query(&format!("SELECT d.id,s.{pk},s.content::text FROM {} s JOIN {} d ON d.key=s.{pk} AND d.catalog=$1 WHERE d.id=ANY($2)",self.table(table),self.table("docs")),&[&catalog,&ids]).map_err(pg_error)?.iter().map(|r|(r.get(0),r.get(1),r.get(2))).collect())
    }
    fn vector_search(
        &self,
        catalog: &str,
        vector: &[f32],
        limit: i64,
    ) -> Result<Vec<(String, String, f64)>> {
        let (table, pk) = catalog_info(catalog)?;
        validate_vector(vector, self.dimensions)?;
        if limit < 0 {
            return Err(invalid("vector search limit must be nonnegative"));
        }
        let vector = Vector::from(vector.to_vec());
        // Bare distance ORDER BY plus LIMIT permits the HNSW access path.
        Ok(self.client.borrow_mut().query(&format!("SELECT {pk},content::text,embedding OPERATOR({ext}.<=>) $1 AS distance FROM {table} WHERE embedding IS NOT NULL ORDER BY embedding OPERATOR({ext}.<=>) $1 LIMIT $2",table=self.table(table),ext=self.extension),&[&vector,&limit]).map_err(pg_error)?.iter().map(|r|(r.get(0),r.get(1),r.get(2))).collect())
    }
    pub fn vector_search_knowledge(
        &self,
        vector: &[f32],
        limit: i64,
    ) -> Result<Vec<(String, String, f64)>> {
        self.vector_search("knowledge", vector, limit)
    }
    pub fn vector_search_statements(
        &self,
        vector: &[f32],
        limit: i64,
    ) -> Result<Vec<(String, String, f64)>> {
        self.vector_search("statement", vector, limit)
    }
    pub fn search(&self, query: &str, opts: &SearchOpts) -> Result<Vec<FtsResult>> {
        if opts.limit < 0 || opts.offset < 0 {
            return Err(invalid("search limit and offset must be nonnegative"));
        }
        if let Some(c) = opts.catalog.as_deref() {
            catalog_info(c)?;
        }
        let segmented = crate::text::segment_for_fts(query);
        let mut quoted = false;
        let query = segmented
            .split_whitespace()
            .filter(|word| {
                let explicit_and = !quoted && *word == "AND";
                if word.chars().filter(|c| *c == '"').count() % 2 == 1 {
                    quoted = !quoted;
                }
                !explicit_and
            })
            .collect::<Vec<_>>()
            .join(" ");
        // PG weights are D,C,B,A and in [0,1]. Preserve 1:3:5:10 and lower-is-better.
        let sql = format!(
            "SELECT d.id,d.catalog,d.key,COALESCE(k.content,s.content)::text AS content,-ts_rank(ARRAY[0.1,0.3,0.5,1.0]::real[],d.search_vector,q.query)::double precision AS rank FROM {docs} d CROSS JOIN websearch_to_tsquery('pg_catalog.simple'::regconfig,$1) AS q(query) LEFT JOIN {knowledge} k ON d.catalog='knowledge' AND k.name=d.key LEFT JOIN {statement} s ON d.catalog='statement' AND s.triple=d.key WHERE d.search_vector @@ q.query AND ($2::text IS NULL OR d.catalog=$2) ORDER BY rank,d.id LIMIT $3 OFFSET $4",
            docs = self.table("docs"),
            knowledge = self.table("knowledge"),
            statement = self.table("statement")
        );
        Ok(self
            .client
            .borrow_mut()
            .query(&sql, &[&query, &opts.catalog, &opts.limit, &opts.offset])
            .map_err(pg_error)?
            .iter()
            .map(|r| FtsResult {
                id: r.get("id"),
                catalog: r.get("catalog"),
                key: r.get("key"),
                content: r.get("content"),
                rank: r.get("rank"),
            })
            .collect())
    }
    pub fn rebuild_indexes(&self) -> Result<()> {
        let mut client = self.client.borrow_mut();
        let mut tx = client.transaction().map_err(pg_error)?;
        tx.batch_execute(&format!(
            "LOCK TABLE {},{},{} IN EXCLUSIVE MODE",
            self.table("knowledge"),
            self.table("statement"),
            self.table("docs")
        ))
        .map_err(pg_error)?;
        for (table, pk) in [("knowledge", "name"), ("statement", "triple")] {
            for row in tx
                .query(
                    &format!("SELECT {pk},content FROM {}", self.table(table)),
                    &[],
                )
                .map_err(pg_error)?
            {
                let key: String = row.get(0);
                let content: Content = serde_json::from_value(row.get(1))?;
                let (_, payload, tokens) = content_values(&content)?;
                tx.execute(
                    &format!(
                        "UPDATE {} SET payload=$2,tokens=$3 WHERE {pk}=$1",
                        self.table(table)
                    ),
                    &[&key, &payload, &tokens],
                )
                .map_err(pg_error)?;
                self.upsert_doc(&mut tx, table, &key, &content)?;
            }
            tx.batch_execute(&format!("REINDEX TABLE {}", self.table(table)))
                .map_err(pg_error)?;
        }
        tx.batch_execute(&format!("REINDEX TABLE {}", self.table("docs")))
            .map_err(pg_error)?;
        tx.commit().map_err(pg_error)
    }
    pub fn snapshot(&self) -> Result<Snapshot> {
        let mut client = self.client.borrow_mut();
        let mut tx = client
            .build_transaction()
            .isolation_level(postgres::IsolationLevel::RepeatableRead)
            .read_only(true)
            .start()
            .map_err(pg_error)?;
        let knowledge = tx
            .query(
                &format!("SELECT * FROM {} ORDER BY name", self.table("knowledge")),
                &[],
            )
            .map_err(pg_error)?
            .iter()
            .map(|r| {
                Ok(KnowledgeRecord {
                    knowledge: knowledge_row(r)?,
                    embedding: r
                        .try_get::<_, Option<Vector>>("embedding")
                        .map_err(pg_error)?
                        .map(|v| v.to_vec()),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let statements = tx
            .query(
                &format!("SELECT * FROM {} ORDER BY triple", self.table("statement")),
                &[],
            )
            .map_err(pg_error)?
            .iter()
            .map(|r| {
                Ok(StatementRecord {
                    statement: statement_row(r)?,
                    embedding: r
                        .try_get::<_, Option<Vector>>("embedding")
                        .map_err(pg_error)?
                        .map(|v| v.to_vec()),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        tx.commit().map_err(pg_error)?;
        Ok(Snapshot {
            format_version: 1,
            embedding: Some(EmbeddingMetadata {
                model: self.model.clone(),
                dimensions: self.dimensions,
                metric: "cosine".into(),
            }),
            knowledge,
            statements,
        })
    }
    pub fn import_snapshot(&self, snapshot: &Snapshot) -> Result<()> {
        snapshot.validate()?;
        for record in &snapshot.knowledge {
            validate_timestamp(record.knowledge.created_at)?;
        }
        for record in &snapshot.statements {
            let s = &record.statement;
            for value in [Some(s.created_at), s.tr_start, s.tr_end]
                .into_iter()
                .flatten()
            {
                validate_timestamp(value)?;
            }
        }
        if let Some(meta) = &snapshot.embedding {
            if meta.model != self.model
                || meta.dimensions != self.dimensions
                || meta.metric != "cosine"
            {
                return Err(invalid(
                    "snapshot embedding metadata does not match target shelf",
                ));
            }
        }
        let mut client = self.client.borrow_mut();
        let mut tx = client.transaction().map_err(pg_error)?;
        tx.batch_execute(&format!(
            "LOCK TABLE {},{},{} IN EXCLUSIVE MODE",
            self.table("knowledge"),
            self.table("statement"),
            self.table("docs")
        ))
        .map_err(pg_error)?;
        let occupied:bool=tx.query_one(&format!("SELECT EXISTS(SELECT 1 FROM {}) OR EXISTS(SELECT 1 FROM {}) OR EXISTS(SELECT 1 FROM {})",self.table("knowledge"),self.table("statement"),self.table("docs")),&[]).map_err(pg_error)?.get(0);
        if occupied {
            return Err(invalid("snapshot import requires an empty target shelf"));
        }
        for record in &snapshot.knowledge {
            let k = &record.knowledge;
            let (raw, payload, tokens) = content_values(&k.content)?;
            let embedding = record.embedding.clone().map(Vector::from);
            tx.execute(&format!("INSERT INTO {}(name,content,payload,created_at,embedding,tokens) VALUES($1,$2,$3,$4,$5,$6)",self.table("knowledge")),&[&k.name,&raw,&payload,&k.created_at,&embedding,&tokens]).map_err(pg_error)?;
            self.upsert_doc(&mut tx, "knowledge", &k.name, &k.content)?;
        }
        for record in &snapshot.statements {
            let s = &record.statement;
            let triple = s.key.to_csv_key();
            let (raw, payload, tokens) = content_values(&s.content)?;
            let embedding = record.embedding.clone().map(Vector::from);
            tx.execute(&format!("INSERT INTO {}(triple,head,relation,tail,content,payload,created_at,tr_start,tr_end,embedding,tokens) VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)",self.table("statement")),&[&triple,&s.key.head,&s.key.relation,&s.key.tail,&raw,&payload,&s.created_at,&s.tr_start,&s.tr_end,&embedding,&tokens]).map_err(pg_error)?;
            self.upsert_doc(&mut tx, "statement", &triple, &s.content)?;
        }
        tx.commit().map_err(pg_error)
    }
}
#[cfg(test)]
mod timestamp_tests {
    use super::*;
    #[test]
    fn rejects_precision_loss_before_database_write() {
        let date = chrono::NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        assert!(validate_timestamp(date.and_hms_nano_opt(0, 0, 0, 123456000).unwrap()).is_ok());
        assert!(validate_timestamp(date.and_hms_nano_opt(0, 0, 0, 123456001).unwrap()).is_err());
    }
}

fn parameter(value: &Value, ty: &Type) -> Result<Box<dyn ToSql + Sync>> {
    macro_rules! typed {
        ($t:ty,$v:expr) => {{
            let v: Option<$t> = if value.is_null() { None } else { Some($v) };
            return Ok(Box::new(v));
        }};
    }
    let bad = || {
        invalid(format!(
            "query parameter cannot be converted to PostgreSQL {}",
            ty.name()
        ))
    };
    match *ty {
        Type::TEXT | Type::VARCHAR | Type::BPCHAR | Type::NAME | Type::UNKNOWN => {
            typed!(String, value.as_str().ok_or_else(bad)?.to_string())
        }
        Type::BOOL => typed!(bool, value.as_bool().ok_or_else(bad)?),
        Type::INT2 => typed!(
            i16,
            i16::try_from(value.as_i64().ok_or_else(bad)?).map_err(|_| bad())?
        ),
        Type::INT4 => typed!(
            i32,
            i32::try_from(value.as_i64().ok_or_else(bad)?).map_err(|_| bad())?
        ),
        Type::INT8 => typed!(i64, value.as_i64().ok_or_else(bad)?),
        Type::FLOAT4 => {
            let n = value.as_f64().unwrap_or(0.0) as f32;
            if !value.is_null() && (!value.is_number() || !n.is_finite()) {
                return Err(bad());
            }
            typed!(f32, n)
        }
        Type::FLOAT8 => typed!(f64, value.as_f64().ok_or_else(bad)?),
        Type::JSON | Type::JSONB => Ok(Box::new(value.clone())),
        Type::TEXT_ARRAY | Type::VARCHAR_ARRAY => typed!(
            Vec<String>,
            value
                .as_array()
                .ok_or_else(bad)?
                .iter()
                .map(|v| v.as_str().map(str::to_string).ok_or_else(bad))
                .collect::<Result<Vec<_>>>()?
        ),
        Type::INT8_ARRAY => typed!(
            Vec<i64>,
            value
                .as_array()
                .ok_or_else(bad)?
                .iter()
                .map(|v| v.as_i64().ok_or_else(bad))
                .collect::<Result<Vec<_>>>()?
        ),
        Type::TIMESTAMP => typed!(
            NaiveDateTime,
            parse_timestamp(value.as_str().ok_or_else(bad)?)?
        ),
        Type::TIMESTAMPTZ => typed!(
            chrono::DateTime<chrono::Utc>,
            chrono::DateTime::parse_from_rfc3339(value.as_str().ok_or_else(bad)?)
                .map_err(|_| bad())?
                .with_timezone(&chrono::Utc)
        ),
        Type::DATE => typed!(
            chrono::NaiveDate,
            chrono::NaiveDate::parse_from_str(value.as_str().ok_or_else(bad)?, "%Y-%m-%d")
                .map_err(|_| bad())?
        ),
        _ => Err(invalid(format!(
            "unsupported PostgreSQL query parameter type: {}",
            ty.name()
        ))),
    }
}
fn parse_timestamp(text: &str) -> Result<NaiveDateTime> {
    NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S%.f")
        .or_else(|_| NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%M:%S%.f"))
        .or_else(|_| chrono::DateTime::parse_from_rfc3339(text).map(|t| t.naive_utc()))
        .map_err(|_| invalid("invalid timestamp query parameter"))
}
