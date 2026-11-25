<#
 iniciar_v2.ps1 - Levanta la app Streamlit de FoodSecurity AI V2 (solo 127.0.0.1) usando el PostgreSQL local
 embebido y los artefactos reales del despliegue V1 (SISTEMA_ALIMENTARIO\despliegue_local).
 Uso:  powershell -ExecutionPolicy Bypass -File .\scripts_operacion\iniciar_v2.ps1 [-Puerto 8501]
 No imprime secretos. Si el despliegue V1 no existe, usa la app sin base de datos (mapas y resultados agregados).
#>
param([int]$Puerto = 8501, [string]$DespliegueV1 = 'despliegue_local')
$ErrorActionPreference = 'Stop'
$Raiz = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$py = (Get-Command python -ErrorAction Stop).Source
$env:PYTHONUTF8 = '1'
New-Item -ItemType Directory -Force -Path (Join-Path $Raiz 'logs') | Out-Null

$envV1 = Join-Path $DespliegueV1 '.env'
if (Test-Path $envV1) {
    Get-Content $envV1 | ForEach-Object {
        if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)=(.*)$') { [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process') }
    }
    Push-Location $DespliegueV1
    & $py preparar.py postgres | Out-Null      # idempotente: si ya esta activo no hace nada
    Pop-Location
} else { Write-Host 'Aviso: no hay .env de V1; la app arrancara sin base de datos.' }

# El codigo es el de V2
$env:FSAI_RAIZ_PROYECTO = Join-Path $Raiz 'foodsecurity_ai'
$appPy = Join-Path $Raiz 'app_streamlit\app\app.py'
$p = Start-Process -FilePath $py -WorkingDirectory (Split-Path -Parent $appPy) -WindowStyle Hidden -PassThru `
     -ArgumentList @('-m','streamlit','run',('"' + $appPy + '"'),'--server.address','127.0.0.1','--server.port',$Puerto,'--server.headless','true','--browser.gatherUsageStats','false') `
     -RedirectStandardOutput (Join-Path $Raiz 'logs\streamlit_v2.out.log') -RedirectStandardError (Join-Path $Raiz 'logs\streamlit_v2.err.log')
Set-Content -Path (Join-Path $Raiz 'logs\streamlit_v2.pid') -Value $p.Id
$url = "http://127.0.0.1:$Puerto"
for ($i = 0; $i -lt 60; $i++) {
    try { if ((Invoke-WebRequest -UseBasicParsing "$url/_stcore/health" -TimeoutSec 2).Content -eq 'ok') { Write-Host "App V2 lista en $url (PID $($p.Id))"; exit 0 } } catch {}
    Start-Sleep -Seconds 1
}
Write-Host 'La app no respondio; revise logs\streamlit_v2.err.log'; exit 1
