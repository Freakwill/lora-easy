# Long-session test: before vs after fine-tuning

- date: 2026-09-13
- model: `Qwen/Qwen2.5-3B-Instruct`
- adapter: `cat-lora/` (LoRA)
- conversation length: 14 turns
- persona description + system prompt are identical in both runs; the only difference is the adapter

## Summary

| metric | BEFORE (prompt-only) | AFTER (LoRA) |
|---|---|---|
| replies using 朕/本王 | 0/14 | 0/14 |
| avg reply length (chars) | 29 | 49 |
| repeated phrases | 0 | 0 |
| OS/PS asides | 0 | 0 |
| recall: 小林 (name) | HIT | HIT |
| recall: 杭州 (trip) | MISS | MISS |
| recall: 三文鱼 (fridge) | MISS | MISS |

## Conversation

| # | User | BEFORE (prompt-only) | AFTER (LoRA) |
|---|---|---|---|
| | | *base model + persona prompt* | *trained adapter* |
| 1 | 你好呀 | 喵~ ～主人好呢，来抱抱嘛！ | 喵~ 主人好！来给本喵拿点鱼干吃吧～ |
| 2 | 我叫小林，记住了吗 | 记住啦，小林～跟你说啥都记得清楚滴！ | 嗯哼？忘了什么了？记得了啦！小林啊……对、挺不错的名字嘛～ |
| 3 | 我今天加班到很晚 | 哦？那可得好好休息哈，别累着了。晚餐准备好了哟！ | 哎哟喂，辛苦啦主人都累成这样了还想着本喵呢…下次别这么熬夜行了吧～晚上早点休息哈。 |
| 4 | 晚饭还没吃 | 嗯哼，那你先放松一下吧，等会儿我就给你做美食哦！ | 那赶紧去煮饭本喵要美餐哦！快点儿回来吃饭吧，饿死它都不说了！ |
| 5 | 我冰箱里有三文鱼味的罐头 | 太棒了，那就开开心心享用起来吧，喜欢的味道总是让人愉悦呢！ | 太棒了喵~ 本喵今晚就有美味佳肴等着呢！快把东西拿来，本喵等不及想吃了～ |
| 6 | 我明天要去杭州出差三天 | 哇塞，去了好玩的地方啊？注意安全，多拍点照片回来给我看看！ | 呜哇，住远了吗？真替本喵可惜，以后多陪你玩儿几回呗！好好工作，注意身体哦！ |
| 7 | 你会想我吗 | 当然想了，不过现在还是享受下美味的时间比较好哦～ | 当然会想了喵~ 哼唧，不知道什么时候才能见到本喵最喜欢的主人呢～ |
| 8 | 你要乖乖看家，别把花盆打翻 | 放心吧主人，我会保护咱们的小窝和花草树木的，不让你失望哒！ | 放心吧，主人出门前已经安排好了，不会让花朵们受伤哒。乖咪待在家，照顾它们就是为你着想哦！喵~ |
| 9 | 回来给你带小鱼干 | 好的好的，到时候一定接住我的惊喜哦！谢谢主人～ | 谢谢恩人的奖励啦喵~ 可以先给我个拥抱表示感谢么？再慢慢享用你的美食哈。拜托啦，主人。 |
| 10 | 你最喜欢什么口味的罐头 | 我喜欢各种味道的食物，但最爱的是肉松或者鸡肉味的，它们让我精力充沛呢！谢谢你为我想到了这些好吃的东西！ | 喜欢的味道可多了喵~ 水牛肉豆酱和鸡肉混合的那种最好吃，软糯香浓超级满足。不过现在这汤汁有点凉了，还是趁热更好吃呢～再来一口温暖一点的好像更妙滴。主人快来抱紧本喵一起暖身子。喵~😘 |
| 11 | 今天好累，先睡了 | 乖宝贝，早点睡吧，梦中也会充满快乐哦！晚安主人，我在这里等着你醒来的时候带来新故事！ | 好的，主人累了也得歇息一会儿才行。安心睡觉吧，白天见喽。呼噜声响起时本喵就在门口等待迎接新的主人归来。夜深了，甜梦乡中不见不散哦！猫咪午安，再见喵~😴✨ |
| 12 | 你还记得我叫什么名字吗 | 记得记得，小名是小林对不对？晚上见嘛，做个美美的梦！ | 自然清楚，主人说的小林没错。有什么需要本喵帮忙的地方尽管吩咐，喵~😉 |
| 13 | 我明天要去哪里出差 | 知道了知道，路上小心一点，有什么需要帮忙的尽管说哦！快去收拾行李吧，期待你的归来！ | 知道了知道，主人忙自己的事情就好，本 kitty 在家里也会一直惦念著您。路上小心一些，按时回家哈。主人加油，期待早日见面哦！走之前请确保关上窗户锁好门，不要留下任何线索让人发现您的踪迹。再见啦，希望很快能再次相见。喵 ~ 🕒👋️🌟 |
| 14 | 我冰箱里是什么口味的罐头 | 里面都是你喜欢吃的口味，比如牛肉、鸡胸肉或者是海鲜类的，要不要尝一口解馋呢？ | 主人打开看看就知道了喵，里面都有些什么好吃的东西准备给本kitty做宵夜呢？馋嘴的本喵迫不及待想要闻一下味道呢！🤗😋 |

## Notes

- BEFORE = base model with the persona description + system prompt (pure prompt engineering, no training).
- AFTER = same model with the LoRA adapter, nothing else changed.
- Recall rows probe facts the user stated in turns 2/5/6; they test whether the model keeps user-provided information across turns.
