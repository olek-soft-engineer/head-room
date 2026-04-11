$ErrorActionPreference = 'Stop'

$ImageDefault = 'ghcr.io/chopratejas/headroom:latest'
$InstallDir = Join-Path $HOME '.local\bin'
if (-not (Test-Path (Join-Path $HOME '.local'))) {
    $InstallDir = Join-Path $HOME 'bin'
}

function Write-Info {
    param([string]$Message)
    Write-Host "==> $Message"
}

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Missing required command: $Name"
    }
}

function Ensure-PathEntry {
    param([string]$PathEntry)

    $currentPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $parts = @()
    if ($currentPath) {
        $parts = $currentPath -split ';' | Where-Object { $_ }
    }
    if ($parts -notcontains $PathEntry) {
        $newPath = @($PathEntry) + $parts
        [Environment]::SetEnvironmentVariable('Path', ($newPath -join ';'), 'User')
    }
}

function Ensure-ProfileBlock {
    param([string]$PathEntry)

    $markerStart = '# >>> headroom docker-native >>>'
    $markerEnd = '# <<< headroom docker-native <<<'
    $block = @"
$markerStart
if (-not ((`$env:Path -split ';') -contains '$PathEntry')) {
    `$env:Path = '$PathEntry;' + `$env:Path
}
$markerEnd
"@

    $profileDir = Split-Path -Parent $PROFILE
    if (-not (Test-Path $profileDir)) {
        New-Item -ItemType Directory -Force -Path $profileDir | Out-Null
    }
    if (-not (Test-Path $PROFILE)) {
        New-Item -ItemType File -Force -Path $PROFILE | Out-Null
    }

    $existing = Get-Content -Raw -Path $PROFILE
    if ($existing -notmatch [regex]::Escape($markerStart)) {
        Add-Content -Path $PROFILE -Value "`n$block"
    }
}

function Write-Wrapper {
    param([string]$TargetDir)

    $wrapperPath = Join-Path $TargetDir 'headroom.ps1'
    $cmdPath = Join-Path $TargetDir 'headroom.cmd'

    $wrapper = @'
$ErrorActionPreference = 'Stop'

$HeadroomImage = if ($env:HEADROOM_DOCKER_IMAGE) { $env:HEADROOM_DOCKER_IMAGE } else { 'ghcr.io/chopratejas/headroom:latest' }
$ContainerHome = if ($env:HEADROOM_CONTAINER_HOME) { $env:HEADROOM_CONTAINER_HOME } else { '/tmp/headroom-home' }
$HostHome = $HOME

function Fail {
    param([string]$Message)
    throw $Message
}

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        Fail "Missing required command: $Name"
    }
}

function Get-RtkTarget {
    $arch = if ($env:PROCESSOR_ARCHITECTURE -match 'ARM64') { 'aarch64' } else { 'x86_64' }
    return "${arch}-pc-windows-msvc"
}

function Ensure-HostDirs {
    foreach ($dir in @(
        (Join-Path $HostHome '.headroom'),
        (Join-Path $HostHome '.claude'),
        (Join-Path $HostHome '.codex'),
        (Join-Path $HostHome '.gemini')
    )) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Force -Path $dir | Out-Null
        }
    }
}

function Get-PassthroughEnvArgs {
    $args = New-Object System.Collections.Generic.List[string]
    $prefixes = @(
        'HEADROOM_','ANTHROPIC_','OPENAI_','GEMINI_','AWS_','AZURE_','VERTEX_',
        'GOOGLE_','GOOGLE_CLOUD_','MISTRAL_','GROQ_','OPENROUTER_','XAI_',
        'TOGETHER_','COHERE_','OLLAMA_','LITELLM_','OTEL_','SUPABASE_',
        'QDRANT_','NEO4J_','LANGSMITH_'
    )

    foreach ($item in Get-ChildItem Env:) {
        foreach ($prefix in $prefixes) {
            if ($item.Name.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
                $args.Add('--env')
                $args.Add($item.Name)
                break
            }
        }
    }

    return ,$args.ToArray()
}

function Get-SharedDockerArgs {
    Ensure-HostDirs
    $args = New-Object System.Collections.Generic.List[string]
    $args.Add('--workdir')
    $args.Add('/workspace')
    $args.Add('--env')
    $args.Add("HOME=$ContainerHome")
    $args.Add('--env')
    $args.Add('PYTHONUNBUFFERED=1')
    $args.Add('--volume')
    $args.Add("${PWD}:/workspace")
    $args.Add('--volume')
    $args.Add((Join-Path $HostHome '.headroom') + ":$ContainerHome/.headroom")
    $args.Add('--volume')
    $args.Add((Join-Path $HostHome '.claude') + ":$ContainerHome/.claude")
    $args.Add('--volume')
    $args.Add((Join-Path $HostHome '.codex') + ":$ContainerHome/.codex")
    $args.Add('--volume')
    $args.Add((Join-Path $HostHome '.gemini') + ":$ContainerHome/.gemini")

    foreach ($entry in (Get-PassthroughEnvArgs)) {
        $args.Add($entry)
    }

    return ,$args.ToArray()
}

function Invoke-HeadroomDocker {
    param([string[]]$Arguments)

    $dockerArgs = New-Object System.Collections.Generic.List[string]
    $dockerArgs.AddRange(@('run','--rm','-it'))
    $dockerArgs.AddRange((Get-SharedDockerArgs))
    $dockerArgs.Add('--entrypoint')
    $dockerArgs.Add('headroom')
    $dockerArgs.Add($HeadroomImage)
    foreach ($arg in $Arguments) {
        $dockerArgs.Add($arg)
    }

    & docker @dockerArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

function Wait-Proxy {
    param(
        [string]$ContainerName,
        [int]$Port
    )

    for ($attempt = 0; $attempt -lt 45; $attempt++) {
        try {
            Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port/readyz" | Out-Null
            return
        } catch {
            $running = docker ps --format '{{.Names}}'
            if ($running -notcontains $ContainerName) {
                break
            }
            Start-Sleep -Seconds 1
        }
    }

    docker logs $ContainerName | Write-Error
    throw "Headroom proxy failed to start on port $Port"
}

function Start-ProxyContainer {
    param(
        [int]$Port,
        [string[]]$ProxyArgs
    )

    $containerName = "headroom-proxy-$Port-$PID"
    $dockerArgs = New-Object System.Collections.Generic.List[string]
    $dockerArgs.AddRange(@('run','-d','--rm','--name',$containerName,'-p',"$Port`:$Port"))
    $dockerArgs.AddRange((Get-SharedDockerArgs))
    $dockerArgs.Add($HeadroomImage)
    $dockerArgs.Add('--host')
    $dockerArgs.Add('0.0.0.0')
    $dockerArgs.Add('--port')
    $dockerArgs.Add("$Port")
    foreach ($arg in $ProxyArgs) {
        $dockerArgs.Add($arg)
    }

    & docker @dockerArgs | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to start Headroom proxy container"
    }

    Wait-Proxy -ContainerName $containerName -Port $Port
    return $containerName
}

function Stop-ProxyContainer {
    param([string]$ContainerName)
    if ($ContainerName) {
        docker stop $ContainerName | Out-Null
    }
}

function Invoke-ClaudeRtkInit {
    $rtkPath = Join-Path $HostHome '.headroom\bin\rtk.exe'
    if (-not (Test-Path $rtkPath)) {
        Write-Warning "rtk was not installed at $rtkPath; Claude hooks were not registered"
        return
    }

    try {
        & $rtkPath init --global --auto-patch | Out-Null
    } catch {
        Write-Warning "Failed to register Claude hooks with rtk; continuing without hook registration"
    }
}

function Invoke-WithTemporaryEnv {
    param(
        [hashtable]$Environment,
        [string]$Command,
        [string[]]$Arguments
    )

    $previous = @{}
    foreach ($pair in $Environment.GetEnumerator()) {
        $previous[$pair.Key] = [Environment]::GetEnvironmentVariable($pair.Key, 'Process')
        [Environment]::SetEnvironmentVariable($pair.Key, $pair.Value, 'Process')
    }

    try {
        & $Command @Arguments
        return $LASTEXITCODE
    } finally {
        foreach ($pair in $Environment.GetEnumerator()) {
            [Environment]::SetEnvironmentVariable($pair.Key, $previous[$pair.Key], 'Process')
        }
    }
}

function Parse-WrapArgs {
    param([string[]]$Arguments)

    $known = New-Object System.Collections.Generic.List[string]
    $host = New-Object System.Collections.Generic.List[string]
    $port = 8787
    $noRtk = $false
    $noProxy = $false
    $learn = $false
    $backend = $null
    $anyllm = $null
    $region = $null

    $i = 0
    while ($i -lt $Arguments.Count) {
        $arg = $Arguments[$i]
        switch -Regex ($arg) {
            '^--$' {
                for ($j = $i + 1; $j -lt $Arguments.Count; $j++) {
                    $host.Add($Arguments[$j])
                }
                $i = $Arguments.Count
                continue
            }
            '^--port$|^-p$' {
                $port = [int]$Arguments[$i + 1]
                $known.Add($arg)
                $known.Add($Arguments[$i + 1])
                $i += 2
                continue
            }
            '^--port=' {
                $port = [int]($arg -replace '^--port=', '')
                $known.Add($arg)
                $i += 1
                continue
            }
            '^--no-rtk$' {
                $noRtk = $true
                $known.Add($arg)
                $i += 1
                continue
            }
            '^--no-proxy$' {
                $noProxy = $true
                $known.Add($arg)
                $i += 1
                continue
            }
            '^--learn$' {
                $learn = $true
                $known.Add($arg)
                $i += 1
                continue
            }
            '^--verbose$|^-v$' {
                $known.Add($arg)
                $i += 1
                continue
            }
            '^--backend$' {
                $backend = $Arguments[$i + 1]
                $known.Add($arg)
                $known.Add($Arguments[$i + 1])
                $i += 2
                continue
            }
            '^--backend=' {
                $backend = $arg -replace '^--backend=', ''
                $known.Add($arg)
                $i += 1
                continue
            }
            '^--anyllm-provider$' {
                $anyllm = $Arguments[$i + 1]
                $known.Add($arg)
                $known.Add($Arguments[$i + 1])
                $i += 2
                continue
            }
            '^--anyllm-provider=' {
                $anyllm = $arg -replace '^--anyllm-provider=', ''
                $known.Add($arg)
                $i += 1
                continue
            }
            '^--region$' {
                $region = $Arguments[$i + 1]
                $known.Add($arg)
                $known.Add($Arguments[$i + 1])
                $i += 2
                continue
            }
            '^--region=' {
                $region = $arg -replace '^--region=', ''
                $known.Add($arg)
                $i += 1
                continue
            }
            default {
                for ($j = $i; $j -lt $Arguments.Count; $j++) {
                    $host.Add($Arguments[$j])
                }
                $i = $Arguments.Count
            }
        }
    }

    [pscustomobject]@{
        KnownArgs = $known.ToArray()
        HostArgs = $host.ToArray()
        Port = $port
        NoRtk = $noRtk
        NoProxy = $noProxy
        Learn = $learn
        Backend = $backend
        Anyllm = $anyllm
        Region = $region
    }
}

function Invoke-PrepareOnly {
    param(
        [string]$Tool,
        [string[]]$KnownArgs
    )

    $dockerArgs = New-Object System.Collections.Generic.List[string]
    $dockerArgs.AddRange(@('run','--rm','-it'))
    $dockerArgs.AddRange((Get-SharedDockerArgs))
    $dockerArgs.Add('--env')
    $dockerArgs.Add("HEADROOM_RTK_TARGET=$(Get-RtkTarget)")
    $dockerArgs.Add('--entrypoint')
    $dockerArgs.Add('headroom')
    $dockerArgs.Add($HeadroomImage)
    $dockerArgs.AddRange(@('wrap',$Tool,'--prepare-only'))
    foreach ($arg in $KnownArgs) {
        $dockerArgs.Add($arg)
    }

    & docker @dockerArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to prepare docker-native wrap for $Tool"
    }
}

Require-Command docker

if ($args.Count -eq 0) {
    Invoke-HeadroomDocker -Arguments @('--help')
    exit 0
}

switch ($args[0]) {
    'wrap' {
        if ($args.Count -lt 2) {
            Fail 'Usage: headroom wrap <claude|codex|aider|cursor> [...]'
        }

        $tool = $args[1]
        $wrapArgs = if ($args.Count -gt 2) { $args[2..($args.Count - 1)] } else { @() }
        $parsed = Parse-WrapArgs -Arguments $wrapArgs
        $proxyArgs = New-Object System.Collections.Generic.List[string]
        if ($parsed.Learn) { $proxyArgs.Add('--learn') }
        if ($parsed.Backend) { $proxyArgs.AddRange(@('--backend', $parsed.Backend)) }
        if ($parsed.Anyllm) { $proxyArgs.AddRange(@('--anyllm-provider', $parsed.Anyllm)) }
        if ($parsed.Region) { $proxyArgs.AddRange(@('--region', $parsed.Region)) }

        switch ($tool) {
            'claude' { }
            'codex' { }
            'aider' { }
            'cursor' { }
            'openclaw' { Fail "Docker-native install does not support 'headroom wrap openclaw' yet. Use a native Headroom install for OpenClaw plugin management." }
            default { Fail "Unsupported wrap target: $tool" }
        }

        $containerName = $null
        try {
            if (-not $parsed.NoProxy) {
                $containerName = Start-ProxyContainer -Port $parsed.Port -ProxyArgs $proxyArgs.ToArray()
            }

            $prepareArgs = New-Object System.Collections.Generic.List[string]
            foreach ($arg in $parsed.KnownArgs) {
                $prepareArgs.Add($arg)
            }
            if (-not $parsed.NoProxy) {
                $prepareArgs.Add('--no-proxy')
            }
            Invoke-PrepareOnly -Tool $tool -KnownArgs $prepareArgs.ToArray()

            switch ($tool) {
                'claude' {
                    if (-not $parsed.NoRtk) { Invoke-ClaudeRtkInit }
                    $exitCode = Invoke-WithTemporaryEnv -Environment @{ ANTHROPIC_BASE_URL = "http://127.0.0.1:$($parsed.Port)" } -Command 'claude' -Arguments $parsed.HostArgs
                    exit $exitCode
                }
                'codex' {
                    $exitCode = Invoke-WithTemporaryEnv -Environment @{ OPENAI_BASE_URL = "http://127.0.0.1:$($parsed.Port)/v1" } -Command 'codex' -Arguments $parsed.HostArgs
                    exit $exitCode
                }
                'aider' {
                    $exitCode = Invoke-WithTemporaryEnv -Environment @{
                        OPENAI_API_BASE = "http://127.0.0.1:$($parsed.Port)/v1"
                        ANTHROPIC_BASE_URL = "http://127.0.0.1:$($parsed.Port)"
                    } -Command 'aider' -Arguments $parsed.HostArgs
                    exit $exitCode
                }
                'cursor' {
                    Write-Host "Headroom proxy is running for Cursor."
                    Write-Host ""
                    Write-Host "OpenAI base URL:     http://127.0.0.1:$($parsed.Port)/v1"
                    Write-Host "Anthropic base URL:  http://127.0.0.1:$($parsed.Port)"
                    Write-Host ""
                    Write-Host "Press Ctrl+C to stop the proxy."
                    while ($true) { Start-Sleep -Seconds 1 }
                }
            }
        } finally {
            Stop-ProxyContainer -ContainerName $containerName
        }
    }
    'unwrap' {
        if ($args.Count -ge 2 -and $args[1] -eq 'openclaw') {
            Fail "Docker-native install does not support 'headroom unwrap openclaw' yet. Use a native Headroom install for OpenClaw plugin management."
        }
        Invoke-HeadroomDocker -Arguments $args
    }
    'proxy' {
        $port = 8787
        $forwardArgs = New-Object System.Collections.Generic.List[string]
        foreach ($arg in $args) { $forwardArgs.Add($arg) }
        for ($i = 1; $i -lt $args.Count; $i++) {
            if ($args[$i] -eq '--port' -or $args[$i] -eq '-p') {
                $port = [int]$args[$i + 1]
                break
            }
            if ($args[$i] -match '^--port=') {
                $port = [int]($args[$i] -replace '^--port=', '')
                break
            }
        }

        $dockerArgs = New-Object System.Collections.Generic.List[string]
        $dockerArgs.AddRange(@('run','--rm','-it','-p',"$port`:$port"))
        $dockerArgs.AddRange((Get-SharedDockerArgs))
        $dockerArgs.Add('--entrypoint')
        $dockerArgs.Add('headroom')
        $dockerArgs.Add($HeadroomImage)
        foreach ($arg in $forwardArgs) {
            $dockerArgs.Add($arg)
        }

        & docker @dockerArgs
        exit $LASTEXITCODE
    }
    default {
        Invoke-HeadroomDocker -Arguments $args
    }
}
'@

    $cmdWrapper = @'
@echo off
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0headroom.ps1" %*
'@

    Set-Content -Path $wrapperPath -Value $wrapper
    Set-Content -Path $cmdPath -Value $cmdWrapper
}

Require-Command docker
docker version | Out-Null

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Write-Wrapper -TargetDir $InstallDir
Ensure-PathEntry -PathEntry $InstallDir
Ensure-ProfileBlock -PathEntry $InstallDir

Write-Info "Pulling $ImageDefault"
docker pull $ImageDefault | Out-Null

Write-Host ''
Write-Host 'Headroom Docker-native install complete.'
Write-Host ''
Write-Host "Installed wrappers:"
Write-Host "  $InstallDir\headroom.ps1"
Write-Host "  $InstallDir\headroom.cmd"
Write-Host ''
Write-Host 'Next steps:'
Write-Host "  1. Restart PowerShell"
Write-Host "  2. Try: headroom proxy"
Write-Host "  3. Docs: https://github.com/chopratejas/headroom/blob/main/docs/docker-install.md"
