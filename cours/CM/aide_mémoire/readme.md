# MD to pdf

setup

````bash
sudo apt update && \
sudo apt install -y texlive texlive-latex-extra texlive-xetex pandoc
````

convert:

````bash
sudo pandoc 01-Nlp\ intro.md -o 01-nlp_intro.pdf
````