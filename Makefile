.PHONY: all install data results quick test paper clean

all: results paper

install:
	pip install -e ".[dev]"

data:
	python -m mpbounds.data

results:            ## full run: estimates, bootstrap (B=499), tables, figures  (~15 min)
	python -m mpbounds.pipeline

quick:              ## same pipeline with B=49 draws
	python -m mpbounds.pipeline --quick

test:
	python -m pytest -q

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -quiet main.tex

clean:
	rm -rf data/processed/* results/* paper/*.aux paper/*.log paper/*.bbl paper/*.blg paper/*.fdb_latexmk paper/*.fls paper/*.out paper/*.toc
