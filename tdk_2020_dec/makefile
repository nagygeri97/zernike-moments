FILE=main

pdf:
	pdflatex $(FILE).tex 

all:
	pdflatex $(FILE).tex
	bibtex $(FILE)
	pdflatex $(FILE).tex
	pdflatex $(FILE).tex

clean:
	rm *.aux
	rm *.bbl
	rm *.blg
	rm *.dvi
	rm *.fdb_latexmk
	rm *.fls
	rm *.log
	rm *.synctex.gz
