if not "%MSMPI_BIN_CONDA_BACKUP%"=="" (
    set "MSMPI_BIN=%MSMPI_BIN_CONDA_BACKUP%"
    set "MSMPI_BIN_CONDA_BACKUP="
) else (
    set "MSMPI_BIN="
)

if not "%MSMPI_INC_CONDA_BACKUP%"=="" (
    set "MSMPI_INC=%MSMPI_INC_CONDA_BACKUP%"
    set "MSMPI_INC_CONDA_BACKUP="
) else (
    set "MSMPI_INC="
)

if not "%MSMPI_LIB64_CONDA_BACKUP%"=="" (
    set "MSMPI_LIB64=%MSMPI_LIB64_CONDA_BACKUP%"
    set "MSMPI_LIB64_CONDA_BACKUP="
) else (
    set "MSMPI_LIB64="
)

if not "%MSMPI_LIB32_CONDA_BACKUP%"=="" (
    set "MSMPI_LIB32=%MSMPI_LIB32_CONDA_BACKUP%"
    set "MSMPI_LIB32_CONDA_BACKUP="
)
