#!/bin/bash
set -e

# DEPENDENCIES
#   poppler
#   paps

[ -d tmp-pdf ] || mkdir tmp-pdf

tree | paps --font='Monospace 8' --header --format=pdf --header-left='{path}' --header-center= --landscape --paper=letter -o tmp-pdf/0.pdf

files=$(find . -name "*.cpp" -o -name "*.h" -o -name "bench*.sh" -o name "Makefile" | sort -r)
i=1
for file in $files; do
  paps --font='Monospace 8' --header --format=pdf --header-left='{path}' --header-center= --landscape --paper=letter $file -o tmp-pdf/$i.pdf
  ((i++))
done

pdfunite $(eval echo tmp-pdf/"{0..$(($i-1))}".pdf) out.pdf
rm -r tmp-pdf
