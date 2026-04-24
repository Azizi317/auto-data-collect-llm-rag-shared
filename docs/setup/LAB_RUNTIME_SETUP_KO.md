# 런타임 설정 안내

현재 목표는 실행 환경을 홈 디렉터리 캐시 중심이 아니라, 저장소 내부 기준으로 최대한 묶는 것이다.

## 1. PowerShell 환경 적용

작업 전에 아래 스크립트를 먼저 불러온다.

```powershell
cd "<프로젝트_폴더>"
. .\scripts\esgreveal_env.ps1
```

이 스크립트는 아래 값을 repo 기준으로 맞춘다.

- `HF_HOME`
- `HUGGINGFACE_HUB_CACHE`
- `TRANSFORMERS_CACHE`
- `SENTENCE_TRANSFORMERS_HOME`
- `MODELSCOPE_CACHE`
- `TORCH_HOME`
- `ESGREVEAL_*` 기본 모델 설정

## 2. 기존 캐시를 repo 안으로 옮기기

이미 받아둔 모델 캐시가 있다면 아래 스크립트로 repo `.cache`로 복사한다.

```powershell
.\scripts\sync_existing_caches.ps1
```

## 3. 모델 사전 다운로드

필요한 모델을 미리 받아둘 수 있다.

```powershell
.\ESGReveal Setup.cmd
.\.venv\Scripts\Activate.ps1
python .\scripts\prefetch_models.py
```

LORE만 건너뛰고 싶으면:

```powershell
python .\scripts\prefetch_models.py --skip-lore
```

## 4. 같이 옮기면 좋은 것

다른 PC로 이동할 때는 아래를 같이 가져가면 재다운로드 시간을 줄일 수 있다.

- 저장소 전체 폴더
- `.venv`
- `.cache`

즉, 현재 기준 실행 핵심은 `repo + .venv + .cache` 조합이다.
