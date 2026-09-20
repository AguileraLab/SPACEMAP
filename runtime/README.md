# runtime/

Scratch space for things the application creates while it runs, when
`SPACEMAP_HOME` points at this checkout rather than at your home directory.

Nothing in here is source code and nothing in here is committed: the directory
exists only so that a fresh clone has somewhere to put runtime files.

By default SPACEMAP keeps all of this outside the repository instead, under
`~/.spacemap/`. Run this to see where your installation is actually writing:

```python
from spacemap.userpaths import describe_layout
print(describe_layout())
```

The local model server's log is written to the `logs` directory shown there.
Start and stop the server from **AI Assistant → Model → Local model**.
