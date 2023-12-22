This site runs on [BookStack](https://www.bookstackapp.com/), a PHP-based Wiki/documentation software. While there are other documentation generators I considered ([mdBook](https://rust-lang.github.io/mdBook/), [MkDocs](https://www.mkdocs.org/)) I wanted something that could allow collaboration/contribution without managing Github pull requests, which didn't leave too many easy/open source options ([HedgeDoc](https://hedgedoc.org/), [Outline](https://github.com/outline/outline), and [Wiki.js](https://js.wiki/) were runners up).

Bookstack is run with Docker Compose using [solidnerd/docker-bookstack](https://github.com/solidnerd/docker-bookstack). (I spent a detour setting up my own container without Apache 2.0 but ran into issues with php-fpm and ended up giving up).


## 2023-12 Review
While Bookstack has worked pretty well, I'm actually not such a big fan of it's rendering (slightly less nice than the other "Doc" style sites) and there's a bit more friction than I'd like.

### Requirements
* Fast Editing
  * Ideally collaboration support but not necessary
* Good Markdown support (ideally in-line editing)
  * Ideally could work w/ git, MarkText, Obsidian etc, but not a requirement
* Navigation (eg Hedgedoc much faster to edit but has no nav)
* Search
* Github logins (for 3rd party contribs)
  * Ideally allow banning/locking in case spam becomes an issue

### Options
* Bookstack - our current winner, just keep running it
* Flowershow or Quartz - skip contributors, just keep this as a personal logbook? Will make editin a lot smoother

### Unsuitable
* Hedgedoc - doesn't support good search, navigation, but very convenient editing
* Outline - slick, but was impossible to setup/lost data before
* Affine, Appflowy, AnyType are neat Notion clones but aren't suitable for publishing
* [OtterWiki](https://github.com/redimp/otterwiki) - neat, but not quite right for this