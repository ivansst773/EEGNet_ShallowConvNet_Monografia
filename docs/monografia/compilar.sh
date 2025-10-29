#!/bin/bash
# Script para compilar presentación Beamer con biblatex + biber

# Nombre base del archivo (sin .tex)
FILE="main"

# Primera pasada de LaTeX
pdflatex -interaction=nonstopmode $FILE.tex

# Procesar bibliografía con biber
biber $FILE

# Dos pasadas más de LaTeX para actualizar referencias
pdflatex -interaction=nonstopmode $FILE.tex
pdflatex -interaction=nonstopmode $FILE.tex

echo "✅ Compilación terminada. Revisa $FILE.pdf"
