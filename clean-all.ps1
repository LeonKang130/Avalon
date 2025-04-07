# Remove all files in the "dataset" directory
Get-ChildItem -Path "dataset" -File | Remove-Item -Force

# Remove all files in the "output" directory
Get-ChildItem -Path "output" -File | Remove-Item -Force

# Remove all files in the "checkpoint" directory
Get-ChildItem -Path "checkpoint" -File | Remove-Item -Force
