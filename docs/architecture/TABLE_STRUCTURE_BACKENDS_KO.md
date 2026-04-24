# Table Structure Backend 안내

현재 표 구조 인식은 아래 순서로 시도한다.

1. `Table Transformer`
2. `LORE-TSR`
3. `PyMuPDF find_tables()` fallback

## 1. Table Transformer

- 공식 repo: [microsoft/table-transformer](https://github.com/microsoft/table-transformer)
- 현재 워크스페이스 위치: `external/table-transformer`
- 역할:
  - 표 영역 내부의 `row`, `column`, `header`, `spanning cell`을 인식
  - 셀 단위 구조 복원
  - 표 텍스트를 다시 `row x column` 형태로 정리

필요한 환경변수:

- `ESGREVEAL_TABLE_TRANSFORMER_MODEL`
- `ESGREVEAL_TABLE_TRANSFORMER_DEVICE`

예시:

```powershell
cd "<프로젝트_폴더>"
.\ESGReveal Setup.cmd
.\.venv\Scripts\Activate.ps1
$env:ESGREVEAL_TABLE_BACKEND="table_transformer,fitz"
$env:ESGREVEAL_TABLE_TRANSFORMER_MODEL="microsoft/table-transformer-structure-recognition-v1.1-all"
python -m esgreveal --pdf "reports\2025_kr\Hyundai_Motor_Sustainability_Report_2025_KOR.pdf" --out table_transformer_test.json
```

## 2. LORE-TSR

- 공식 repo: [AlibabaResearch/AdvancedLiterateMachinery - LORE-TSR](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/DocumentUnderstanding/LORE-TSR)
- 공유본에는 로컬 repo를 포함하지 않고, 기본적으로 `ModelScope` 모델 경로를 사용한다
- 역할:
  - 선이 약한 표 구조를 복원
  - adapter 또는 ModelScope pipeline 결과를 공통 형식으로 정규화

필요한 환경변수:

- `ESGREVEAL_LORE_PATH`
- `ESGREVEAL_LORE_PREDICTOR`
- 또는 `ESGREVEAL_LORE_MODEL`

예시:

```powershell
cd "<프로젝트_폴더>"
.\ESGReveal Setup.cmd
.\.venv\Scripts\Activate.ps1
$env:ESGREVEAL_TABLE_BACKEND="lore_tsr,fitz"
$env:ESGREVEAL_LORE_MODEL="damo/cv_resnet-transformer_table-structure-recognition_lore"
python -c "from esgreveal.table_structure_analysis import _lore_pipeline_bundle; print(bool(_lore_pipeline_bundle('damo/cv_resnet-transformer_table-structure-recognition_lore')))"
```

참고:

- 현재 CPU 환경에서는 ModelScope LORE 구현 내부의 `cuda()` 호출을 우회해서 실행한다.
- 이 우회는 [table_structure_analysis.py](../../esgreveal/table_structure_analysis.py)에 반영되어 있다.

## 3. 현재 구현 방식

- [table_structure_analysis.py](../../esgreveal/table_structure_analysis.py)에서 세 backend를 하나의 공통 형식으로 묶는다.
- [document.py](../../esgreveal/document.py)에서는 `table.extract()` 결과만 믿지 않고, 가능하면 표 bbox를 crop해서 구조 인식을 먼저 시도한다.
- 구조 인식이 성공하면 셀 기반 텍스트를 우선 사용하고, 실패하면 기존 추출 결과를 fallback으로 사용한다.
