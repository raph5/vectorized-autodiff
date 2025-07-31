#!/usr/bin/env bash
set -e

# DEPENDENCIES
#   poppler
#   paps

[ -d tmp-pdf ] || mkdir tmp-pdf

tree | paps --font='Monospace 8' --header --format=pdf --header-left='{path}' --header-center= --landscape --paper=letter -o tmp-pdf/4by3-0.pdf
tree | paps --font='Monospace 8' --header --format=pdf --header-left='{path}' --header-center= --paper=A4 -o tmp-pdf/A4-0.pdf

files=$(find . -name "*.cpp" -o -name "*.h" -o -name "bench*.sh" -o -name "Makefile" | sort -r)
i=1
for file in $files; do
  paps --font='Monospace 8' --header --format=pdf --header-left='{path}' --header-center= --landscape --paper=letter $file -o tmp-pdf/4by3-$i.pdf
  paps --font='Monospace 8' --header --format=pdf --header-left='{path}' --header-center= --paper=A4 $file -o tmp-pdf/A4-$i.pdf
  ((i++))
done

pdfunite $(eval echo tmp-pdf/4by3-"{0..$(($i-1))}".pdf) 4by3.pdf
pdfunite $(eval echo tmp-pdf/A4-"{0..$(($i-1))}".pdf) A4.pdf
rm -r tmp-pdf
