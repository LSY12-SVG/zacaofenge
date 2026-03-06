param(
  [string]$Logdir = "runs",
  [int]$Port = 6006,
  [string]$Host = "127.0.0.1"
)

Write-Host "Starting TensorBoard..."
Write-Host "  logdir: $Logdir"
Write-Host "  host:   $Host"
Write-Host "  port:   $Port"

tensorboard --logdir "$Logdir" --port $Port --host $Host
