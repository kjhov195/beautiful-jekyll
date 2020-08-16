---
layout: post
title: Sublime Text-setting
subtitle: initial setting
category: Dev
use_math: true
---


# Package Control

Tools > Install Package Control

<br>

### sublimeREPL

Ctrl + Shift + P > install package > sublimeREPL

<br>

### SideBarEnhancements

Ctrl + Shift + P > install package > SideBarEnhancements

<br>

### SideBarEnhancements

Ctrl + Shift + P > install package > Anaconda

<br>

### MarkdownPreview

Ctrl + Shift + P > install package > MarkdownPreview

Preferences > Package Settings > Markdown Preview > Settings User

```
{
    "enable_mathjax": true, "html_simple": false,
}
```

<br>

# Setting

Preferences > Key Bindings

```
[
    {"keys": ["ctrl+shift+m"],
    "command": "markdown_preview", "args": {"target": "browser", "parser":"markdown"} 
    },

    {
        "keys": ["f5"],
       "command": "repl_open",
                     "caption": "Python - RUN current file",
                     "id": "repl_python_run",
                     "mnemonic": "R",
                     "args": {
                        "type": "subprocess",
                        "encoding": "utf8",
                        "cmd": ["python", "-u", "$file_basename"],
                        "cwd": "$file_path",
                        "syntax": "Packages/Python/Python.tmLanguage",
                        "external_id": "python",
                        "extend_env": {"PYTHONIOENCODING": "utf-8"}
                    }
    }
]
```

<br>

Preferences > Settings

```
{
    "font_face": "나눔고딕코딩",
    "font_size": 12,
    "highlight_line": true,
    "ignored_packages":
    [
        "Vintage"
    ],
    "tab_size": 4,
    "translate_tabs_to_spaces": true
}
```

<br>
<br>