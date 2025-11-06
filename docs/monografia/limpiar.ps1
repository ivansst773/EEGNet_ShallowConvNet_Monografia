# limpiar.ps1
# Script para borrar archivos auxiliares de LaTeX en la carpeta actual

$extensiones = @(
    "*.aux", "*.bbl", "*.bcf", "*.blg",
    "*.fdb_latexmk", "*.fls", "*.log",
    "*.nav", "*.out", "*.run.xml",
    "*.snm", "*.toc"
)

foreach ($ext in $extensiones) {
    Get-ChildItem -Path . -Include $ext -Recurse -ErrorAction SilentlyContinue | Remove-Item -Force
}

Write-Host "✅ Archivos auxiliares eliminados. Listo para recompilar."
