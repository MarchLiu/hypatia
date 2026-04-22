# Memory

1. 用户每发送对话检查一次，如果内容包括用户明确的提及hypatia，进行下一步检查
2. 如果用户明确的提及记住、遗忘、和对记忆的修改，进入记忆提取环节
3. 否则每10轮，或者在session结束、或者session compact的时候，才触发记忆提取
4. 记忆提取时除了提取内容，还要建立必要的关系
5. 对于需要遗忘的消息，也要删除相关的知识条目和条目的关系
6. 对于用户明确提及的肯定规则，或者在session中反复出现的模式，记忆为 rule，对于否定规则，记忆为 tabo
7. hypatia的kowledge和statemewnt中，每一条数据包含名为 scope 的字符串数组，在meta中记录收集到的这条知识时涉及的project，记录到scopes中，如果用户明确提及，或者AI整理出了某个可以作用于全局的内容，在scope中加入一个名为"”的字符串项
8. 新建立—个session的时候，从匹配的项目 （session的项目加上"）中load规则和禁忌
9. 每12小时整理一次hypatia的default库，review新加入的session与既往知识是否存在可以添加的关联