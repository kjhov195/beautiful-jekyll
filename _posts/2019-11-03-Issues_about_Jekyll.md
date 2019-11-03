---
layout: post
title: Issues about Jekyll
subtitle: How to use, Errors
category: Jekyll
use_math: true
---
# Result on local
Local에서 결과를 보고 싶을 때
```
bundle exec jekyll server
bundle exec jekyll server start
```

# Encoding Error
Windows에서 error가 날 때 Windows cmd에서
```
c/Windows/System32/chcp.com 65001
```

# Using Mathjax

1) include 폴더에 mathjax_support.html를 다음의 내용으로 작성  

```
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    TeX: {
      equationNumbers: {
        autoNumber: "AMS"
      }
    },
    tex2jax: {
    inlineMath: [ ['$', '$'] ],
    displayMath: [ ['$$', '$$'] ],
    processEscapes: true,
  }
});
MathJax.Hub.Register.MessageHook("Math Processing Error",function (message) {
	  alert("Math Processing Error: "+message[1]);
	});
MathJax.Hub.Register.MessageHook("TeX Jax - parse error",function (message) {
	  alert("Math Processing Error: "+message[1]);
	});
</script>
<script type="text/javascript" async
  charset="utf-8"
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
```

2. Mathjax를 사용할 post에서 다음 옵션을 사용  

```
use_math: true
```
