# Rules
- Only fix the files I requested.
- Think before if a code change should be put in a new file/folder.
- Each file should not have more than 300 lines. Try not to clog existing files
- Don't write tests just to pass them.
- Keep working even if there are changes outside of yours

## Web
- Do not be overly verbose. Only add what I asked, don't try to add subtitles, captions,... to everything I asked you to build.
- Do not add a "Live" container or chip unless explicitly asked.

## Python Development Best Practices
### Ignore Python 2 compatibility
This project uses Python 3+. You should not use the __future__ module.

If you need to worry about feature compatibility between different 3.xx point releases, check the closest pyproject.toml's requires-python field to see what minimum runtime version is supported.

### Platform Support
Tests and features must support Linux, macOS and Windows unless feature is explicitly OS-specific.

Codex supports running connected app-server and exec-server on different operating systems. See the $remote-tests skill for details about integration testing these configurations.