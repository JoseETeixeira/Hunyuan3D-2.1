# Detect the host NVIDIA GPU's CUDA compute capability and build the image with the
# native CUDA extensions (custom_rasterizer + mesh painter) compiled for exactly
# that architecture, then start the stack. Docker build has no GPU access, so the
# arch must be detected here on the host and passed in as a build arg.
#
# Usage:
#   ./build.ps1                 # -> docker compose up --build
#   ./build.ps1 build           # -> docker compose build
#   ./build.ps1 up -d --build   # forward any compose args
$ErrorActionPreference = "Stop"

function Map-NameToArch([string]$name) {
    $n = $name.ToLower()
    if ($n -match 'rtx 50|5090|5080|5070|5060|b200|blackwell')            { return "12.0" }  # Blackwell
    if ($n -match 'rtx 40|4090|4080|4070|4060|ada|l40|l4')                { return "8.9"  }  # Ada Lovelace
    if ($n -match 'a100|a800')                                            { return "8.0"  }  # Ampere (datacenter)
    if ($n -match 'rtx 30|3090|3080|3070|3060|a6000|a40|a10|ampere')      { return "8.6"  }  # Ampere (consumer)
    if ($n -match 'rtx 20|2080|2070|2060|titan rtx|t4|gtx 16|1660|1650|turing') { return "7.5" }  # Turing
    if ($n -match 'v100')                                                 { return "7.0"  }  # Volta
    if ($n -match 'gtx 10|1080|1070|1060|pascal')                         { return "6.1"  }  # Pascal
    return $null
}

function Get-ArchList {
    $list = @()
    # Primary: nvidia-smi reports compute capability directly (driver 510+), e.g. "8.6", "12.0".
    try { $caps = & nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null } catch { $caps = $null }
    foreach ($c in $caps) {
        $c = "$c".Trim()
        if ($c -match '^\d+\.\d+$') { $list += $c }
    }
    if ($list.Count -eq 0) {
        # Fallback for older drivers without compute_cap: map GPU model name -> arch.
        try { $names = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null } catch { $names = @() }
        foreach ($n in $names) {
            $a = Map-NameToArch "$n"
            if ($a) { $list += $a }
        }
    }
    return ($list | Select-Object -Unique)
}

$archs = Get-ArchList
if (-not $archs -or @($archs).Count -eq 0) {
    Write-Warning "Could not detect GPU compute capability via nvidia-smi. Defaulting to 12.0 (RTX 50-series)."
    $archs = @("12.0")
}
$env:TORCH_CUDA_ARCH_LIST = (@($archs) -join ";")
Write-Host "[build] Detected GPU arch(s): $($env:TORCH_CUDA_ARCH_LIST)" -ForegroundColor Green

$composeArgs = if ($args.Count -gt 0) { $args } else { @("up", "--build") }
Write-Host "[build] docker compose $($composeArgs -join ' ')  (TORCH_CUDA_ARCH_LIST=$($env:TORCH_CUDA_ARCH_LIST))" -ForegroundColor Cyan
& docker compose @composeArgs
exit $LASTEXITCODE
