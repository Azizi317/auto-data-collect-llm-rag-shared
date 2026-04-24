# GeoLayoutLM Adapter 안내

이 문서는 현재 워크스페이스에서 `GeoLayoutLM`을 전처리 backend로 연결하는 방식을 설명한다.

## 1. 현재 기본 동작

기본 adapter 진입점은 아래다.

`esgreveal.adapters.geolayoutlm_workspace:predict_layout`

이 adapter는 다음 순서로 동작한다.

1. 페이지 이미지를 OCR로 단어/라인 단위로 분해한다.
2. GeoLayoutLM이 요구하는 입력 형식으로 변환한다.
   - `input_ids`
   - `bbox_4p_normalized`
   - `first_token_idxes`
   - `block_mask`
   - `line_rank_id`
   - `line_rank_inner_id`
3. GeoLayoutLM SER checkpoint로 라벨링을 수행한다.
4. `HEADER` 계열은 `heading`, 나머지는 `paragraph`로 변환한다.
5. 표는 GeoLayoutLM이 직접 담당하지 않고 기존 표 검출 경로가 계속 담당한다.

즉, 이 adapter는 외부 repo를 단순 호출하는 수준이 아니라, 우리 PDF 페이지를 GeoLayoutLM 입력 형식으로 바꾸는 workspace 전용 연결층이다.

## 2. 필요한 환경변수

- `ESGREVEAL_LAYOUT_BACKEND`
  - 예: `geolayoutlm,layoutlmv3,layoutparser`
- `ESGREVEAL_GEOLAYOUTLM_PATH`
  - 기본값: `external/AdvancedLiterateMachinery/DocumentUnderstanding/GeoLayoutLM`
- `ESGREVEAL_GEOLAYOUTLM_CHECKPOINT`
  - GeoLayoutLM SER checkpoint 파일 경로
- `ESGREVEAL_GEOLAYOUTLM_CONFIG_JSON`
  - 기본값: `configs/GeoLayoutLM/GeoLayoutLM_large_model_config.json`
- `ESGREVEAL_GEOLAYOUTLM_DEVICE`
  - 예: `cuda`, `cpu`
- `ESGREVEAL_GEOLAYOUTLM_OCR_LANG`
  - 기본값: `kor+eng`

선택 옵션:

- `ESGREVEAL_GEOLAYOUTLM_MAX_SEQ_LENGTH`
- `ESGREVEAL_GEOLAYOUTLM_MAX_BLOCK_NUM`
- `ESGREVEAL_GEOLAYOUTLM_IMG_H`
- `ESGREVEAL_GEOLAYOUTLM_IMG_W`

## 3. 예시

```powershell
$root = (Get-Location).Path
$env:ESGREVEAL_LAYOUT_BACKEND="geolayoutlm,layoutlmv3,layoutparser"
$env:ESGREVEAL_GEOLAYOUTLM_PATH=Join-Path $root "external\AdvancedLiterateMachinery\DocumentUnderstanding\GeoLayoutLM"
$env:ESGREVEAL_GEOLAYOUTLM_CHECKPOINT="C:\models\geolayoutlm\epoch.105-f1_labeling.0.9232.pt"
$env:ESGREVEAL_GEOLAYOUTLM_DEVICE="cuda"
```

## 4. 한계

- 공개 GeoLayoutLM repo는 FUNSD용 VIE 코드라서 `heading/paragraph/table` 전용 구조 모델은 아니다.
- 현재 adapter는 SER 결과 중 `HEADER`를 heading 신호로 활용하는 방식이다.
- 표는 이 adapter가 직접 분류하지 않고, 별도 표 구조 분석 경로와 함께 동작한다.

그래도 이 방식의 장점은, GeoLayoutLM 본체와 그 입력 구조를 실제 전처리 파이프라인에 연결했다는 점이다.
