# Disposable execution sandbox for run_shell_command / run_python_code.
# It carries NO host secrets (no .env, no ~/.ssh) and only the data dir is
# mounted (read-only) at runtime. A full shell is safe here because the
# container — not a command denylist — is the security boundary.
FROM python:3.13-slim

# Tools the model actually reached for (PDF extraction, fast grep) + certs.
# libgomp1: OpenMP runtime required by sklearn estimators (HistGradientBoosting,
#   etc.) -- without it model fits crash with "libgomp.so.1: cannot open ...".
# tzdata / tzdata-legacy: the IANA timezone database INCLUDING the legacy US/*
#   aliases (US/Central, US/Eastern, ...) that python:slim strips -- models use
#   those names for UTC->local market-time conversion and otherwise fall back to
#   wrong fixed offsets (DST-incorrect). tzdata-legacy may not exist on all bases,
#   so it's best-effort; the pip `tzdata` below is the guaranteed fallback.
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
        ripgrep \
        ca-certificates \
        curl \
        libgomp1 \
        tzdata \
        coinor-ipopt \
    && (apt-get install -y --no-install-recommends tzdata-legacy || true) \
    && rm -rf /var/lib/apt/lists/*

# Data-analysis / modeling stack so run_python_code works without host deps.
# tzdata + pytz: timezone support -- `tzdata` is the zoneinfo fallback that
#   resolves the legacy US/* names (verified to fix ZoneInfo('US/Central')); `pytz`
#   for code that imports it directly. These prevent silently-wrong DST workarounds.
# xgboost / lightgbm / holidays: ML + holiday-feature libs models reached for.
# pdfplumber / xlrd / fastparquet: extra PDF / .xls / parquet readers.
RUN pip install --no-cache-dir \
        numpy \
        pandas \
        scipy \
        scikit-learn \
        statsmodels \
        numpy-financial \
        geopandas \
        seaborn \
        plotly \
        plotly-geo \
        kaleido \
        pymupdf \
        openpyxl \
        lxml \
        beautifulsoup4 \
        python-dateutil \
        tabulate \
        requests \
        httpx \
        exa_py \
        psycopg2-binary \
        pyomo \
        tzdata \
        pytz \
        xgboost \
        lightgbm \
        holidays \
        pdfplumber \
        xlrd \
        fastparquet

# Run as an unprivileged user; /work is the only writable area at runtime.
RUN useradd --create-home --uid 1000 sandbox
USER sandbox
WORKDIR /work
