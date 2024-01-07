This site was started in early 2023 and is maintained by [Leonard Lin](https://leonardlin.com/)  as a way to track his exploration of LLMs and generative AI.

The contents of the site is now an [obsidian-git](https://github.com/denolehov/obsidian-git) backed Obsidian notebook and pull requests for content can be submitted to the [vault repo](https://github.com/AUGMXNT/llm-tracker.info-vault/).

For publishing, we use [Quartz](https://quartz.jzhao.xyz/). It's very similar to [Flowershow](https://flowershow.app/) but a bit faster, more active, and speedy.


|  | [Flowershow](https://flowershow.app) | [Quartz](https://quartz.jzhao.xyz/) |
| ---- | ---- | ---- |
| License | MIT | MIT |
| Github | https://github.com/datopian/flowershow | https://github.com/jackyzha0/quartz |
| Contributors | https://github.com/datopian/flowershow/graphs/contributors<br>* 2 main contributors<br>* 15 total | https://github.com/jackyzha0/quartz/graphs/contributors<br>* 1 main contributor<br>* 100 contributors (most 1 commit) |
| Docs | https://flowershow.app/docs | https://quartz.jzhao.xyz/ |
| Roadmap | https://github.com/orgs/datopian/projects/45/views/5 |  |
| Stack | node.js<br>npx<br>nextjs<br>tailwind<br>mdx | node.js<br>npx<br>preact |
| Install | - There is a Vercel deployment<br>- [Command-line publishing](https://flowershow.app/docs/publish-tutorial) being deprecated (!?)<br>- `npx flowershow@latest install` -  installs into `.flowershow` folder | - https://quartz.jzhao.xyz/#-get-started<br>- Clone or fork repo |
| Publish | `npx flowershow@latest export` | `npx quartz build` |
| Publish Time | 16s | 2.6s |
| Theming | https://flowershow.app/docs/custom-theme | https://quartz.jzhao.xyz/layout |
| Config |  |  |
| Navigation |  |  |
| Navigation |  |  |
| Graph View |  |  |
| Markup |  |  |
| Weirdness | Does not seem to pick up changes to `config.mjs` in the content folder? | When generating, the `public` folder, although it has the same inode appears empty when directly mounted in docker (for Caddy). This can be worked around by mounting the parent folder |
Testing update
## TODO:
* Add Comments
	* https://flowershow.app/docs/comments

Questions:
* Can we do redirects with Vercel?
	* We need to edit the router...
	* We should figure out existing URLs
	* And auto search for redirects

LLM List
https://llm.extractum.io/list/lm.extractum.io/list/tum.io/list//list//