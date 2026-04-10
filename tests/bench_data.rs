//! Deterministic synthetic data generator for Hypatia benchmarks.
//!
//! Inspired by MemPalace's `PalaceDataGenerator`. Uses seeded RNG for
//! reproducibility. Generates knowledge entries, statement triples,
//! planted needles (for recall measurement), and search queries.

use rand::prelude::*;

// ── Scale configurations (aligned with MemPalace) ─────────────────────

#[derive(Debug, Clone, Copy)]
pub struct ScaleConfig {
    pub n_knowledge: usize,
    pub n_statements: usize,
    pub n_needles: usize,
    pub n_queries: usize,
}

impl ScaleConfig {
    pub fn small() -> Self {
        Self { n_knowledge: 1_000, n_statements: 2_000, n_needles: 20, n_queries: 40 }
    }
    pub fn medium() -> Self {
        Self { n_knowledge: 10_000, n_statements: 20_000, n_needles: 50, n_queries: 100 }
    }
    pub fn large() -> Self {
        Self { n_knowledge: 50_000, n_statements: 100_000, n_needles: 100, n_queries: 200 }
    }

    pub fn from_name(name: &str) -> Self {
        match name {
            "medium" => Self::medium(),
            "large" => Self::large(),
            _ => Self::small(),
        }
    }
}

// ── Vocabulary banks ──────────────────────────────────────────────────

const TECH_TERMS: &[&str] = &[
    "authentication", "authorization", "middleware", "endpoint", "REST API",
    "GraphQL", "WebSocket", "database migration", "ORM", "query optimization",
    "caching strategy", "load balancer", "rate limiting", "pagination",
    "serialization", "validation", "error handling", "logging framework",
    "monitoring", "deployment pipeline", "CI/CD", "containerization",
    "microservice", "event sourcing", "message queue", "pub/sub",
    "connection pooling", "session management", "token refresh", "CORS",
    "SSL termination", "health check", "circuit breaker", "retry logic",
    "batch processing", "stream processing", "data pipeline", "ETL",
    "feature flag", "A/B testing", "blue-green deployment", "canary release",
    "vector database", "embedding model", "full text search", "inverted index",
    "knowledge graph", "triple store", "semantic query", "structured data",
];

const ENTITY_NAMES: &[&str] = &[
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi",
    "Ivan", "Judy", "Karl", "Linda", "Mike", "Nina", "Oscar", "Pat",
    "Quinn", "Rita", "Steve", "Tina", "Ursula", "Victor", "Wendy", "Xander",
];

const PREDICATES: &[&str] = &[
    "works_on", "manages", "reports_to", "collaborates_with",
    "created", "maintains", "uses", "depends_on", "replaced",
    "reviewed", "deployed", "tested", "documented", "mentors", "leads",
    "contributes_to", "is_a", "contains", "references", "integrates_with",
];

const TAGS: &[&str] = &[
    "backend", "frontend", "api", "database", "auth", "testing", "docs",
    "config", "deployment", "models", "performance", "security", "monitoring",
    "infrastructure", "benchmark", "rust", "python", "typescript", "go",
];

const NEEDLE_TOPICS: &[&str] = &[
    "Fibonacci sequence optimization uses memoization with O(n) space complexity",
    "PostgreSQL vacuum autovacuum threshold set to 50 percent for table users",
    "Redis cluster failover timeout configured at 30 seconds with sentinel monitoring",
    "Kubernetes horizontal pod autoscaler targets 70 percent CPU utilization",
    "GraphQL subscription uses WebSocket transport with heartbeat interval 25 seconds",
    "JWT token rotation policy requires refresh every 15 minutes with sliding window",
    "Elasticsearch index sharding strategy uses 5 primary shards with 1 replica each",
    "Docker multi-stage build reduces image size from 1.2GB to 180MB for production",
    "Apache Kafka consumer group rebalance timeout set to 45 seconds",
    "MongoDB change streams resume token persisted every 100 operations",
    "gRPC streaming uses bidirectional flow control with 64KB window size",
    "Prometheus alerting rule fires when p99 latency exceeds 500ms for 5 minutes",
    "Terraform state locking uses DynamoDB with consistent reads enabled",
    "Nginx rate limiting configured at 100 requests per second with burst of 50",
    "SQLAlchemy connection pool size set to 20 with max overflow of 10 connections",
    "React concurrent mode uses startTransition for non-urgent state updates",
    "AWS Lambda cold start mitigation uses provisioned concurrency of 10 instances",
    "Git bisect automated with custom test script for regression hunting",
    "OpenTelemetry trace sampling rate set to 10 percent in production environment",
    "Celery worker prefetch multiplier set to 1 for fair task distribution",
];

// ── Generated data types ──────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct KnowledgeEntry {
    pub name: String,
    pub data: String,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct StatementEntry {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub data: String,
}

#[derive(Debug, Clone)]
pub struct Needle {
    #[allow(dead_code)]
    pub id: String,
    pub name: String,
    pub content: String,
    pub tags: Vec<String>,
    /// The search query that should find this needle.
    pub query: String,
}

#[derive(Debug, Clone)]
pub struct SearchQuery {
    pub query: String,
    pub expected_name: Option<String>,
    pub is_needle: bool,
}

// ── Generator ─────────────────────────────────────────────────────────

/// Remove characters that FTS5 would interpret as query operators.
fn sanitize_fts_query(query: &str) -> String {
    // Replace FTS5 special characters with spaces, then collapse whitespace
    // FTS5 specials: : (column filter), " (phrase), * (prefix), ^ (beginning)
    // + (AND), - (NOT), ( ) (grouping), . (syntax error in some contexts)
    let sanitized: String = query.chars()
        .map(|c| {
            matches!(c, ':' | '"' | '\'' | '*' | '^' | '+' | '-' | '(' | ')' | '.').then_some(' ').unwrap_or(c)
        })
        .collect();
    let mut result = String::new();
    let mut prev_space = false;
    for c in sanitized.chars() {
        if c == ' ' {
            if !prev_space {
                result.push(c);
            }
            prev_space = true;
        } else {
            result.push(c);
            prev_space = false;
        }
    }
    result.trim().to_string()
}

pub struct BenchDataGenerator {
    rng: StdRng,
    config: ScaleConfig,
    pub knowledge: Vec<KnowledgeEntry>,
    pub statements: Vec<StatementEntry>,
    pub needles: Vec<Needle>,
    pub queries: Vec<SearchQuery>,
}

impl BenchDataGenerator {
    pub fn new(config: ScaleConfig) -> Self {
        let rng = StdRng::seed_from_u64(42);
        Self {
            rng,
            config,
            knowledge: Vec::new(),
            statements: Vec::new(),
            needles: Vec::new(),
            queries: Vec::new(),
        }
    }

    pub fn generate(&mut self) {
        self.generate_needles();
        self.generate_knowledge();
        self.generate_statements();
        self.generate_queries();
    }

    fn pick<'a, T>(&mut self, slice: &'a [T]) -> &'a T {
        let idx = self.rng.random_range(0..slice.len());
        &slice[idx]
    }

    fn pick_n<'a, T>(&mut self, slice: &'a [T], n: usize) -> Vec<&'a T> {
        let mut indices: Vec<usize> = (0..slice.len()).collect();
        indices.shuffle(&mut self.rng);
        indices.iter().take(n).map(|&i| &slice[i]).collect()
    }

    fn random_tags(&mut self) -> Vec<String> {
        let n = self.rng.random_range(1..=4);
        self.pick_n(TAGS, n).iter().map(|s| s.to_string()).collect()
    }

    fn random_sentence(&mut self) -> String {
        let n_terms = self.rng.random_range(3..=6);
        let terms: Vec<&str> = self.pick_n(TECH_TERMS, n_terms).iter().map(|&t| *t).collect();
        let entity = *self.pick(ENTITY_NAMES);
        let templates = [
            format!("{entity} discussed {} implementation with the team, focusing on {} patterns and {} best practices.", terms[0], terms.get(1).unwrap_or(&"design"), terms.get(2).unwrap_or(&"testing")),
            format!("The {} module was refactored to improve {} performance. Key change: {} pipeline now uses {} for better throughput.", terms[0], terms.get(1).unwrap_or(&"system"), terms.get(2).unwrap_or(&"data"), terms.get(3).unwrap_or(&"async")),
            format!("Bug report: {} fails when {} is null. Root cause identified as missing {} validation. Fixed by adding {} checks in the {} layer.", terms[0], terms.get(1).unwrap_or(&"input"), terms.get(2).unwrap_or(&"type"), terms.get(3).unwrap_or(&"boundary"), terms.get(4).unwrap_or(&"service")),
            format!("Architecture decision: migrated from {} to {} for {} reasons. Performance improved by {}% after switching to {}-based {}.", terms[0], terms.get(1).unwrap_or(&"new system"), terms.get(2).unwrap_or(&"scalability"), self.rng.random_range(10..80), terms.get(3).unwrap_or(&"event"), terms.get(4).unwrap_or(&"processing")),
            format!("Meeting notes: discussed {} with {entity}. Agreed to implement {} using {} approach. Deadline set for next sprint. Follow-up on {} integration required.", terms[0], terms.get(1).unwrap_or(&"feature"), terms.get(2).unwrap_or(&"modular"), terms.get(3).unwrap_or(&"system")),
        ];
        templates[self.rng.random_range(0..templates.len())].clone()
    }

    fn random_content(&mut self, min_words: usize, max_words: usize) -> String {
        let target = self.rng.random_range(min_words..=max_words);
        let mut sentences = Vec::new();
        let mut count = 0;
        while count < target {
            let s = self.random_sentence();
            count += s.split_whitespace().count();
            sentences.push(s);
        }
        sentences.join(" ")
    }

    fn generate_needles(&mut self) {
        for i in 0..self.config.n_needles {
            let topic = NEEDLE_TOPICS[i % NEEDLE_TOPICS.len()];
            let needle_id = format!("NEEDLE_{i:04}");
            let name = format!("needle_{i:04}");

            // Extract a query from the topic — capture more context for better FTS matching.
            // Truncate at secondary prepositions/conjunctions to keep queries specific but longer.
            let raw_query = if let Some(pos) = topic.find(" with ") {
                topic[..pos].to_string()
            } else if let Some(pos) = topic.find(" for ") {
                topic[..pos].to_string()
            } else if let Some(pos) = topic.find(" from ") {
                topic[..pos].to_string()
            } else if let Some(pos) = topic.find(" in ") {
                topic[..pos].to_string()
            } else {
                topic.chars().take(80).collect()
            };

            // Sanitize for FTS5: remove chars that could be parsed as FTS operators
            // (: is column filter, - is NOT, " is phrase, etc.)
            let query = sanitize_fts_query(&raw_query);

            let content = format!("{needle_id}: {topic}. This is a unique planted needle for recall benchmarking at scale.");

            self.needles.push(Needle {
                id: needle_id,
                name,
                content,
                tags: vec!["benchmark".to_string(), "needle".to_string()],
                query,
            });
        }
    }

    fn generate_knowledge(&mut self) {
        // First insert needles as knowledge entries
        for needle in &self.needles {
            self.knowledge.push(KnowledgeEntry {
                name: needle.name.clone(),
                data: needle.content.clone(),
                tags: needle.tags.clone(),
            });
        }

        // Fill remaining with random entries
        let remaining = self.config.n_knowledge - self.needles.len();
        for i in 0..remaining {
            let _terms: Vec<&str> = self.pick_n(TECH_TERMS, 2).iter().map(|&t| *t).collect();
            let name = format!("knowledge_{i:06}");
            let data = self.random_content(30, 100);
            let tags = self.random_tags();
            self.knowledge.push(KnowledgeEntry { name, data, tags });
        }

        // Shuffle so needles aren't all at the beginning
        self.knowledge.shuffle(&mut self.rng);
    }

    fn generate_statements(&mut self) {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        while self.statements.len() < self.config.n_statements {
            let subject = (*self.pick(ENTITY_NAMES)).to_string();
            let mut object = (*self.pick(ENTITY_NAMES)).to_string();
            while object == subject {
                object = (*self.pick(ENTITY_NAMES)).to_string();
            }
            let predicate = (*self.pick(PREDICATES)).to_string();
            let key = format!("{subject}|{predicate}|{object}");
            if seen.contains(&key) {
                continue;
            }
            seen.insert(key);
            let data = if self.rng.random_bool(0.3) {
                self.random_content(5, 20)
            } else {
                String::new()
            };
            self.statements.push(StatementEntry {
                subject,
                predicate,
                object,
                data,
            });
        }
    }

    fn generate_queries(&mut self) {
        // Half needle queries (known answers for recall)
        let n_needle_queries = self.config.n_queries / 2;
        for needle in self.needles.iter().take(n_needle_queries) {
            self.queries.push(SearchQuery {
                query: needle.query.clone(),
                expected_name: Some(needle.name.clone()),
                is_needle: true,
            });
        }

        // Half generic queries (measure latency only)
        let n_generic = self.config.n_queries - n_needle_queries;
        for _ in 0..n_generic {
            let t1 = *self.pick(TECH_TERMS);
            let t2 = *self.pick(TECH_TERMS);
            self.queries.push(SearchQuery {
                query: format!("{t1} {t2}"),
                expected_name: None,
                is_needle: false,
            });
        }

        self.queries.shuffle(&mut self.rng);
    }
}

// ── Stats helper ──────────────────────────────────────────────────────

pub struct LatencyStats {
    pub p50_us: u64,
    pub p99_us: u64,
    pub max_us: u64,
    pub min_us: u64,
}

impl LatencyStats {
    pub fn from_durations(durations: &[std::time::Duration]) -> Self {
        let mut us: Vec<u64> = durations.iter().map(|d| d.as_micros() as u64).collect();
        us.sort();
        let p50 = us[us.len() / 2];
        let p99 = us[us.len() * 99 / 100];
        let max = *us.last().unwrap_or(&0);
        let min = *us.first().unwrap_or(&0);
        Self { p50_us: p50, p99_us: p99, max_us: max, min_us: min }
    }
}
