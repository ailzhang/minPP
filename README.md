# Pipeline Parallelism for Minimalists

Every paper seems to present pipeline schedules in a slightly different setup, which makes it hard to compare them side by side. So I went down the rabbit hole and built a tiny simulator to explore them in a unified way.

It’s a lightweight, CPU-only tool that lets you:

- Mix and match partitioning, assignment, and execution strategies
- Simulate microbatch flows step by step
- Visualize all schedules in a consistent timeline format

The whole thing is under 1,000 lines of code—not production-ready, a bit rough around the edges—but it helped me understand and compare schedules much more clearly.

📝 Check out the full blog post here:
https://ailzhang.github.io/posts/pipeline-parallelism-demystified/


## Supported schedules

- gpipe
![Gpipe schedule](./plots/gpipe.png)
- 1f1b
![1F1B schedule](./plots/1f1b.png)
- zero bubble 1f1b
![ZB1F1B schedule](./plots/zb1f1b.png)
- eager 1f1b
![Eager 1F1B schedule](./plots/eager1f1b.png)
- interleaved virtual pipeline
![Interleaved virtual pipeline schedule](./plots/interleaved.png)
- vshape zero bubble
![Vshape zb schedule](./plots/vshape_zb.png)
- bfs looping
![BFS looping schedule](./plots/looped_bfs.png)
- dualpipev
![Dualpipev schedule](./plots/dualpipev.png)

## Bug reports

This repo started as a toy side project without thorough and there will definitely be rough edges. PRs and bug reports are welcome! 
