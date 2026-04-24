$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$launcher = Join-Path $root "ESGReveal GUI.cmd"
$icon = Join-Path $root "assets\esgreveal_icon.ico"
$shortcutPath = Join-Path $root "ESGReveal GUI.lnk"

if (-not (Test-Path $launcher)) {
    throw "Launcher not found: $launcher"
}

$wsh = New-Object -ComObject WScript.Shell
$shortcut = $wsh.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $launcher
$shortcut.Arguments = ''
$shortcut.WorkingDirectory = $root
if (Test-Path $icon) {
    $shortcut.IconLocation = $icon
}
$shortcut.WindowStyle = 1
$shortcut.Description = "Launch ESGReveal GUI"
$shortcut.Save()

Write-Host "Created shortcut: $shortcutPath"
