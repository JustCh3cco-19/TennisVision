REPORT = AI_LAB_Report_Template_ACSAI_2025_2026
DIAGRAM = src/figures/02-PipelineDiagram

.PHONY: all report diagram clean

all: diagram report

report: pdf/$(REPORT).pdf

pdf/$(REPORT).pdf: src/$(REPORT).tex src/references.bib src/figures/*.png
	cd src && pdflatex -interaction=nonstopmode $(REPORT)
	cd src && bibtex $(REPORT)
	cd src && pdflatex -interaction=nonstopmode $(REPORT)
	cd src && pdflatex -interaction=nonstopmode $(REPORT)
	mkdir -p pdf
	mv src/$(REPORT).pdf pdf/

# the diagram is a standalone TikZ figure, included in the report as PNG
diagram: $(DIAGRAM).png

$(DIAGRAM).png: $(DIAGRAM).tex
	cd src/figures && pdflatex -interaction=nonstopmode 02-PipelineDiagram
	pdftoppm -png -r 300 -singlefile $(DIAGRAM).pdf $(DIAGRAM)

clean:
	rm -f src/*.aux src/*.log src/*.bbl src/*.blg src/*.out src/*.toc \
		src/figures/*.aux src/figures/*.log
