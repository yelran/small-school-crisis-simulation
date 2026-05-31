import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update, ALL
import dash_leaflet as dl
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import json, os, math
import glob
import plotly.graph_objects as go
from prophet import Prophet
import logging

logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

HIGH_RISK_FILE  = os.path.join(DATA_DIR, "(결과)고위험학교.csv")
LIME_FILE       = os.path.join(DATA_DIR, "LIME_분석.csv")
REC_FILE        = os.path.join(DATA_DIR, "추천결과.csv")
CONSOL_FILE     = os.path.join(DATA_DIR, "(결과)통폐합_시뮬.csv")
INTEG_FILE      = os.path.join(DATA_DIR, "(결과)통합운영_시뮬.csv")
MAINT_FILE      = os.path.join(DATA_DIR, "(결과)분교유지_시뮬.csv")
COORD_FILE      = os.path.join(DATA_DIR, "전국학교_위경도_정제.csv")
CLUSTER_FILE    = os.path.join(DATA_DIR, "(결과)고위험학교_군집분석.csv")
STUDENT_TS_FILE = os.path.join(DATA_DIR, "2008_2025_초등_국공립_학급학생수교사수_통합.csv")

def safe_read(path, **kw):
    if not os.path.exists(path):
        return pd.DataFrame()
    for enc in ('utf-8-sig', 'cp949', 'utf-8'):
        try:
            return pd.read_csv(path, encoding=enc, **kw)
        except (UnicodeDecodeError, LookupError):
            continue
    return pd.DataFrame()


# ── 초등학교 학생수 시계열 로드 (Prophet용: 통합 CSV 1개 사용) ────────────────
STUDENT_TS_FILE = DATA_DIR + r"\데이터\2008_2025_초등_국공립_학급학생수교사수_통합.csv"

print("학생수 시계열 로드 중...")

df_student_ts = safe_read(STUDENT_TS_FILE)

if not df_student_ts.empty:


    # 필요한 컬럼만 사용
    need_cols = ["연도", "시도교육청", "교육지원청", "지역", "학교코드", "학교명", "학생수"]
    df_student_ts = df_student_ts[[c for c in need_cols if c in df_student_ts.columns]].copy()

    # 타입 정리
    df_student_ts["연도"] = pd.to_numeric(df_student_ts["연도"], errors="coerce")
    df_student_ts["학생수"] = pd.to_numeric(df_student_ts["학생수"], errors="coerce")

    for col in ["시도교육청", "교육지원청", "지역", "학교코드", "학교명"]:
        if col in df_student_ts.columns:
            df_student_ts[col] = df_student_ts[col].astype(str).str.strip()

    # 결측 제거
    df_student_ts = df_student_ts.dropna(subset=["연도", "학교명", "학생수"])

    # 같은 학교코드-연도 중복 방지
    if "학교코드" in df_student_ts.columns:
        df_student_ts = (
            df_student_ts
            .groupby(["학교코드", "시도교육청", "교육지원청", "지역", "학교명", "연도"], as_index=False)["학생수"]
            .median()
        )
    else:
        df_student_ts = (
            df_student_ts
            .groupby(["시도교육청", "교육지원청", "지역", "학교명", "연도"], as_index=False)["학생수"]
            .median()
        )

    df_student_ts = df_student_ts.sort_values(["학교명", "연도"]).reset_index(drop=True)

print("학생수 시계열 shape:", df_student_ts.shape)
print(f"  완료: {df_student_ts['학교명'].nunique() if not df_student_ts.empty else 0}개 학교명")


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin(math.radians(lat2 - lat1) / 2) ** 2 + \
        math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


df_high = safe_read(HIGH_RISK_FILE)

if not df_student_ts.empty:
    year_count = df_student_ts.groupby('학교명')['연도'].nunique()
    few_data_schools = year_count[year_count <= 3].index.tolist()
    df_high = df_high[~df_high['학교명'].isin(few_data_schools)]


df_cluster = safe_read(CLUSTER_FILE)
if not df_cluster.empty:
    df_high = df_high.merge(
        df_cluster[['학교명', '군집명']],
        on='학교명', how='left'
    )

df_high = df_high.drop_duplicates(subset=['학교명', '시도교육청', '위도', '경도'], keep='first').reset_index(drop=True)
df_high['_uid'] = df_high.apply(lambda r: f"{r['학교명']}__{r['위도']:.6f}_{r['경도']:.6f}", axis=1)

df_lime = safe_read(LIME_FILE)
df_consol = safe_read(CONSOL_FILE)
df_integ = safe_read(INTEG_FILE)
df_maint = safe_read(MAINT_FILE)
df_coord = safe_read(COORD_FILE)
if not df_coord.empty and '시도교육청명' in df_coord.columns:
    df_coord = df_coord.rename(columns={'시도교육청명': '시도교육청'})

if not df_consol.empty:
    df_consol = df_consol.drop_duplicates(subset=['고위험학교명', '후보학교명'], keep='first').copy()
    df_consol['후보순위'] = df_consol.groupby('고위험학교명').cumcount() + 1
if not df_integ.empty:
    _integ_name_col = '통합후보_학교명' if '통합후보_학교명' in df_integ.columns else '후보학교명'
    df_integ = df_integ.drop_duplicates(subset=['고위험학교명', _integ_name_col], keep='first').copy()
    df_integ['후보순위'] = df_integ.groupby('고위험학교명').cumcount() + 1

SIDO_MAP = {
    '강원특별자치도교육청': '강원특별자치도',
    '전라남도교육청': '전라남도',
    '전북특별자치도교육청': '전북특별자치도',
    '경상북도교육청': '경상북도',
    '충청남도교육청': '충청남도',
    '충청북도교육청': '충청북도',
    '경기도교육청': '경기도',
    '경상남도교육청': '경상남도',
    '인천광역시교육청': '인천광역시',
    '제주특별자치도교육청': '제주특별자치도',
    '부산광역시교육청': '부산광역시',
    '광주광역시교육청': '광주광역시',
}
df_high['시도명'] = df_high['시도교육청'].map(SIDO_MAP).fillna(df_high['시도교육청'])
df_high['시군구'] = df_high['지역'].apply(lambda x: str(x).split()[-1] if pd.notna(x) else '')

FEAT_KR = {
    '교원 1인당 학생수': '교원 1인당 학생수가 매우 적어 소규모 학교 특성 뚜렷',
    '연간 운영비': '연간 운영비가 낮아 재정 압박 가능성',
    '지역 내 학생수 점유율': '지역 내 학생수 점유율이 매우 낮음',
    '학생수 변화율': '학생수 감소 추세가 심각',
    '학생 1인당 운영비': '학생 1인당 운영비 부담이 높음',
    '최근 3년 학생수 감소량': '최근 3년간 학생수가 지속 감소',
    '이중감소 신호': '학생수와 지역 학령인구가 동시 감소',
    '학생수': '학생수가 매우 적음',
    '학급당 학생수': '학급당 학생수가 낮아 소규모 학급 운영',
}


def risk_color(prob):
    if prob >= 0.70: return '#e53e3e'
    return '#d97706'


def risk_label(prob):
    if prob >= 0.70: return '고위험'
    return '중위험'


def card_s():
    return {'background': 'white', 'borderRadius': '12px', 'padding': '16px 20px',
            'boxShadow': '0 1px 4px rgba(0,0,0,0.08)'}


def section_s():
    return {'background': 'white', 'borderRadius': '12px', 'padding': '20px', 'boxShadow': '0 1px 4px rgba(0,0,0,0.08)'}


KOREA_BOUNDS = [[32.0, 123.5], [39.8, 132.8]]
MAP_MIN_ZOOM = 7

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        'https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap'
    ],
    suppress_callback_exceptions=True
)
app.title = "소규모 학교 위기 예측 시뮬레이터"

app.layout = html.Div([
    dcc.Store(id='sel-school', data=None),
    dcc.Store(id='sel-cand', data=None),
    dcc.Store(id='open-det', data=None),
    dcc.Store(id='scroll-to-cand', data=None),
    dcc.Store(id='show-all-cands', data=False),

    html.Div([
        html.Div([
            html.Span("소규모 학교 통폐합 영향 시뮬레이터", style={'fontSize': '16px', 'fontWeight': '700', 'color': '#1a202c'}),
        ], style={'display': 'flex', 'alignItems': 'center'}),
        html.Div("기준 연도: 2025년 | 출처: 학교알리미 · KESS · 교육부 공공데이터",
                 style={'fontSize': '12px', 'color': '#718096', 'marginTop': '4px'}),
    ], style={'padding': '14px 24px', 'background': 'white', 'borderBottom': '1px solid #e2e8f0', 'position': 'sticky',
              'top': 0, 'zIndex': 1000}),

    html.Div([
        html.Div([
            html.Div([
                html.Div("학교 필터",
                         style={'fontWeight': '700', 'fontSize': '13px', 'color': '#2d3748', 'marginBottom': '16px'}),
                html.Div("시·도", style={'fontSize': '12px', 'color': '#718096', 'marginBottom': '4px'}),
                dcc.Dropdown(id='sido-dd',
                             options=[{'label': '전체', 'value': '전체'}] +
                                     [{'label': v, 'value': k} for k, v in SIDO_MAP.items() if
                                      k in df_high['시도교육청'].values],
                             value='전체', clearable=False, style={'fontSize': '13px', 'marginBottom': '12px'}),
                html.Div("시·군·구", style={'fontSize': '12px', 'color': '#718096', 'marginBottom': '4px'}),
                dcc.Dropdown(id='sigungu-dd',
                             options=[{'label': '전체', 'value': '전체'}],
                             value='전체', clearable=False, style={'fontSize': '13px', 'marginBottom': '16px'}),
                html.Hr(style={'margin': '12px 0', 'borderColor': '#e2e8f0'}),
                html.Div(id='school-list'),
            ], style={'padding': '16px'}),
        ], style={'width': '240px', 'minWidth': '240px', 'borderRight': '1px solid #e2e8f0', 'background': 'white',
                  'height': 'calc(100vh - 58px)', 'overflowY': 'auto', 'flexShrink': 0}),

        html.Div([
            dcc.Tabs(id='tabs', value='tab-map', children=[
                dcc.Tab(label='전국 분포 지도', value='tab-map'),
                dcc.Tab(label='학교별 분석', value='tab-analysis'),
            ]),
            html.Div(id='tab-content', style={'height': 'calc(100vh - 100px)', 'overflowY': 'auto'}),
        ], style={'flex': '1', 'overflow': 'hidden'}),
    ], style={'display': 'flex', 'height': 'calc(100vh - 58px)'}),

], style={'fontFamily': "'Noto Sans KR', sans-serif", 'background': '#f7fafc', 'minHeight': '100vh'})


# ── 콜백 ──────────────────────────────────────────────────

@app.callback(Output('sigungu-dd', 'options'), Output('sigungu-dd', 'value'), Input('sido-dd', 'value'))
def upd_sigungu(sido):
    if sido == '전체':
        return [{'label': '전체', 'value': '전체'}], '전체'
    filtered = df_high[df_high['시도교육청'] == sido]
    opts = [{'label': '전체', 'value': '전체'}] + [{'label': s, 'value': s} for s in sorted(filtered['시군구'].unique())]
    return opts, '전체'


@app.callback(Output('school-list', 'children'),
              Input('sido-dd', 'value'), Input('sigungu-dd', 'value'), Input('sel-school', 'data'))
def upd_list(sido, sigungu, selected):
    filtered = df_high.copy()
    if sido != '전체':
        filtered = filtered[filtered['시도교육청'] == sido]
    if sigungu != '전체':
        filtered = filtered[filtered['시군구'] == sigungu]
    if filtered.empty:
        return html.Div("해당 지역에 학교가 없습니다.", style={'fontSize': '12px', 'color': '#a0aec0'})

    items = [html.Div(f"학교 목록 ({len(filtered)}개)",
                      style={'fontSize': '12px', 'fontWeight': '700', 'color': '#4a5568', 'marginBottom': '8px'})]
    for _, row in filtered.iterrows():
        is_sel = (row['_uid'] == selected)
        prob = row['위기확률']
        color = risk_color(prob)
        items.append(html.Div(
            key=row['_uid'],
            id={'type': 'sch', 'index': row['_uid']},
            n_clicks=0,
            children=[
                html.Div(html.Span(row['학교명'],
                                   style={'fontSize': '13px', 'fontWeight': '600' if is_sel else '400'})),
                html.Div([
                    html.Span(f"{row['시군구']} · {int(row['학생수'])}명", style={'fontSize': '11px', 'color': '#718096'}),
                    html.Span(f" {risk_label(prob)}", style={'fontSize': '10px', 'color': color,
                                                             'background': color + '20', 'padding': '1px 6px',
                                                             'borderRadius': '10px', 'marginLeft': '4px'}),
                ]),
            ],
            style={'padding': '8px 10px', 'borderRadius': '8px', 'cursor': 'pointer', 'marginBottom': '4px',
                   'background': '#ebf8ff' if is_sel else 'transparent',
                   'borderLeft': f'3px solid {color}' if is_sel else '3px solid transparent'}
        ))
    return items


@app.callback(
    Output('open-det', 'data'),
    Input({'type': 'det-btn', 'index': ALL}, 'n_clicks'),
    Input('sel-school', 'data'),
    State({'type': 'det-btn', 'index': ALL}, 'id'),
    State('open-det', 'data')
)
def open_detail(clicks, selected, ids, cur):
    ctx = callback_context

    if not ctx.triggered:
        return no_update

    # 학교가 실제로 선택되었을 때만 기본 탭을 통폐합으로 설정
    if ctx.triggered[0]['prop_id'] == 'sel-school.data':
        if selected:
            return '통폐합'
        return no_update

    # 버튼 클릭이 아닌 초기 렌더링은 무시
    if not clicks or all((n is None or n == 0) for n in clicks):
        return no_update

    trig = ctx.triggered_id

    if isinstance(trig, dict):
        return trig['index']

    return no_update


@app.callback(
    Output('sel-school', 'data'),
    Input({'type': 'sch', 'index': ALL}, 'n_clicks'),
    State({'type': 'sch', 'index': ALL}, 'id'),
    prevent_initial_call=True
)
def select_school(clicks, ids):
    ctx = callback_context

    if not ctx.triggered or not clicks or not ids:
        return no_update

    trig = ctx.triggered_id

    if not isinstance(trig, dict):
        return no_update

    # 실제 클릭된 항목의 n_clicks가 1 이상일 때만 선택 처리
    for n, item_id in zip(clicks, ids):
        if item_id == trig and n and n > 0:
            return trig['index']

    # n_clicks=0인 초기 렌더링/자동 트리거는 무시
    return no_update


@app.callback(Output('sel-cand', 'data'),
              Input({'type': 'cand', 'index': ALL}, 'n_clicks'),
              State({'type': 'cand', 'index': ALL}, 'id'),
              State('sel-cand', 'data'), prevent_initial_call=True)
def sel_cand_cb(clicks, ids, cur):
    ctx = callback_context
    if not ctx.triggered:
        return no_update
    triggered_id = ctx.triggered_id
    if triggered_id is None:
        return no_update
    try:
        idx = triggered_id['index']
        return None if idx == cur else idx
    except Exception:
        return no_update


@app.callback(
    Output('sel-cand', 'data', allow_duplicate=True),
    Input({'type': 'cand-map', 'index': ALL}, 'n_clicks'),
    State({'type': 'cand-map', 'index': ALL}, 'id'),
    prevent_initial_call=True
)
def select_cand_from_map(clicks, ids):
    ctx = callback_context
    if not ctx.triggered:
        return no_update
    trig = ctx.triggered_id
    if isinstance(trig, dict):
        return trig['index']
    return no_update


@app.callback(
    Output('sel-cand', 'data', allow_duplicate=True),
    Input({'type': 'integ-map', 'index': ALL}, 'n_clicks'),
    State({'type': 'integ-map', 'index': ALL}, 'id'),
    prevent_initial_call=True
)
def select_integ_from_map(clicks, ids):
    ctx = callback_context
    if not ctx.triggered:
        return no_update
    trig = ctx.triggered_id
    if isinstance(trig, dict):
        return trig['index']
    return no_update


@app.callback(
    Output('show-all-cands', 'data'),
    Input('toggle-all-cands', 'value'),
    prevent_initial_call=True
)
def toggle_all_cands(value):
    return 'show' in value


@app.callback(
    Output('sel-cand', 'data', allow_duplicate=True),
    Input({'type': 'rank-item', 'index': ALL}, 'n_clicks'),
    State({'type': 'rank-item', 'index': ALL}, 'id'),
    prevent_initial_call=True
)
def select_cand_from_rank(clicks, ids):
    ctx = callback_context
    if not ctx.triggered:
        return no_update
    trig = ctx.triggered_id
    if isinstance(trig, dict):
        return trig['index']
    return no_update


@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value'),
    Input('sel-school', 'data'),
    Input('open-det', 'data'),
    Input('sel-cand', 'data'),
    Input('show-all-cands', 'data')
)
def render_tab_callback(tab, selected, open_det, sel_cand, show_all):
    return render_tab(tab, selected, open_det, sel_cand, show_all)


def render_tab(tab, selected, open_det, sel_cand, show_all=False):
    if tab == 'tab-map':
        return render_map(selected)
    return render_analysis(selected, open_det, sel_cand, show_all)


def _lime_chart(school_name):
    if df_lime.empty:
        return html.Span("LIME 분석 결과 파일이 없습니다.", style={'color': '#a0aec0', 'fontSize': '13px'})
    rows = df_lime[df_lime['학교명'] == school_name]
    if rows.empty:
        return html.Span("해당 학교의 LIME 데이터가 없습니다.", style={'color': '#a0aec0', 'fontSize': '13px'})
    lr = rows.iloc[0]

    items = []
    for rank in [1, 2, 3]:
        feat = lr.get(f'위험증가_{rank}위_피처', '')
        val = lr.get(f'위험증가_{rank}위_기여도', None)
        if pd.notna(feat) and feat and pd.notna(val):
            items.append((str(feat), float(val)))
    for rank in [1, 2, 3]:
        feat = lr.get(f'위험감소_{rank}위_피처', '')
        val = lr.get(f'위험감소_{rank}위_기여도', None)
        if pd.notna(feat) and feat and pd.notna(val):
            items.append((str(feat), float(val)))

    if not items:
        return html.Span("기여도 데이터가 없습니다.", style={'color': '#a0aec0', 'fontSize': '13px'})

    merged = {}
    for feat, val in items:
        merged[feat] = merged.get(feat, 0) + val
    items = [(feat, val) for feat, val in merged.items() if abs(val) > 1e-12]
    items.sort(key=lambda x: abs(x[1]))

    max_pos = max([v for _, v in items if v > 0], default=1)
    max_neg = max([abs(v) for _, v in items if v < 0], default=1)
    bar_max_w = 430
    chart_w = bar_max_w * 2 + 2

    rows_html = []
    for feat, val in items:
        side_max = max_pos if val >= 0 else max_neg
        bar_w = max(3, int(abs(val) / side_max * bar_max_w))
        is_pos = val >= 0
        color = '#e53e3e' if is_pos else '#3b82f6'
        label = f'+{val:.3f}' if is_pos else f'{val:.3f}'

        rows_html.append(
            html.Div([
                html.Div(feat, style={'width': '220px', 'textAlign': 'right', 'paddingRight': '16px',
                                      'fontSize': '13px', 'color': '#1f2937', 'fontWeight': '600',
                                      'lineHeight': '1.35', 'whiteSpace': 'normal'}),
                html.Div([
                    html.Div(
                        html.Div(style={'width': f'{bar_w}px' if not is_pos else '0px', 'height': '22px',
                                        'background': '#3b82f6', 'borderRadius': '4px 0 0 4px'}),
                        style={'width': f'{bar_max_w}px', 'display': 'flex', 'justifyContent': 'flex-end'}
                    ),
                    html.Div(style={'width': '2px', 'height': '30px', 'background': '#64748b'}),
                    html.Div(
                        html.Div(style={'width': f'{bar_w}px' if is_pos else '0px', 'height': '22px',
                                        'background': '#e53e3e', 'borderRadius': '0 4px 4px 0'}),
                        style={'width': f'{bar_max_w}px', 'display': 'flex', 'justifyContent': 'flex-start'}
                    ),
                ], style={'width': f'{chart_w}px', 'display': 'flex', 'alignItems': 'center'}),
                html.Div(label, style={'width': '78px', 'paddingLeft': '14px', 'fontSize': '13px',
                                       'fontWeight': '700', 'color': color, 'textAlign': 'left'}),
            ], style={'display': 'grid', 'gridTemplateColumns': f'220px {chart_w}px 78px',
                      'alignItems': 'center', 'justifyContent': 'center', 'columnGap': '0px',
                      'marginBottom': '12px', 'width': '100%'})
        )

    legend = html.Div([
        html.Div(),
        html.Div([
            html.Div("■ 위험 감소 요인", style={'width': f'{bar_max_w}px', 'textAlign': 'center',
                                          'color': '#3b82f6', 'fontSize': '13px', 'fontWeight': '700'}),
            html.Div(style={'width': '2px'}),
            html.Div("■ 위험 증가 요인", style={'width': f'{bar_max_w}px', 'textAlign': 'center',
                                          'color': '#e53e3e', 'fontSize': '13px', 'fontWeight': '700'}),
        ], style={'width': f'{chart_w}px', 'display': 'flex', 'alignItems': 'center'}),
        html.Div(),
    ], style={'display': 'grid', 'gridTemplateColumns': f'220px {chart_w}px 78px',
              'justifyContent': 'center', 'alignItems': 'center', 'marginBottom': '18px'})

    return html.Div([legend,
                     html.Div(rows_html, style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'})
                     ], style={'padding': '20px'})


def render_map(selected):
    markers = []
    selected_markers = []

    # 선택값이 없거나 유효하지 않으면 선택 없음으로 처리
    if not selected or selected not in df_high['_uid'].values:
        selected = None

    selected_row = df_high[df_high['_uid'] == selected] if selected else pd.DataFrame()

    map_center = [36.5, 127.8]
    map_zoom = 7

    if selected and not selected_row.empty:
        map_center = [selected_row.iloc[0]['위도'], selected_row.iloc[0]['경도']]
        map_zoom = 8

    for _, row in df_high.iterrows():
        prob = row['위기확률']
        color = risk_color(prob)
        is_sel = bool(selected) and (row['_uid'] == selected)
        popup = dl.Popup(html.Div([
            html.B(row['학교명']), html.Br(),
            html.Span(row['지역'], style={'fontSize': '12px', 'color': '#666'}), html.Br(),
            html.Span(f"학생수: {int(row['학생수'])}명 | 위기확률: {prob:.1%}", style={'fontSize': '12px'}),
        ]))
        if is_sel:
            selected_markers.extend([
                dl.CircleMarker(center=[row['위도'], row['경도']], radius=15, color=color,
                                weight=3, fillColor=color, fillOpacity=0.16),
                dl.CircleMarker(center=[row['위도'], row['경도']], radius=9, color=color,
                                weight=3, fillColor=color, fillOpacity=0.95, children=popup)
            ])
        else:
            markers.append(dl.CircleMarker(
                center=[row['위도'], row['경도']],
                radius=5,
                color='white',
                weight=1,
                fillColor=color,
                fillOpacity=0.75,
                children=popup
            ))

    sido_count = df_high['시도명'].value_counts()
    max_cnt = sido_count.max()
    sido_bars = []
    for sido, cnt in sido_count.items():
        color = '#e53e3e' if cnt == max_cnt else '#d97706'
        sido_bars.append(html.Div([
            html.Span(f"{sido} {cnt}개", style={'fontSize': '13px', 'color': '#2d3748', 'fontWeight': '600'}),
        ], style={'marginBottom': '10px'}))

    return html.Div([html.Div([
        html.Div([
            html.H5("소규모 학교 위기 분포", style={'fontWeight': '700', 'marginBottom': '4px'}),
            html.P("학생수 감소 추세 + 졸업생 수 + 지역 인구 감소율 → 위험도 종합 산출",
                   style={'fontSize': '13px', 'color': '#718096', 'margin': 0}),
        ], style={'marginBottom': '16px'}),
        html.Div([
            html.Div([
                dl.Map(children=[dl.TileLayer(
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png")] + markers + selected_markers,
                       center=map_center, zoom=map_zoom, minZoom=MAP_MIN_ZOOM, maxBounds=KOREA_BOUNDS,
                       style={'height': '520px', 'borderRadius': '12px', 'overflow': 'hidden'}, id='main-map'),
                html.Div([
                    html.Span("● 고위험", style={'color': '#e53e3e', 'marginRight': '12px', 'fontSize': '12px'}),
                    html.Span("● 중위험", style={'color': '#d97706', 'fontSize': '12px'}),
                ], style={'marginTop': '8px'}),
            ], style={'flex': '2'}),
            html.Div([
                html.Div([
                    html.Div("시도별 분포", style={'fontSize': '14px', 'fontWeight': '700', 'marginBottom': '16px',
                                              'color': '#2d3748'}),
                    *sido_bars,
                ], style={**section_s(), 'marginBottom': '16px'}),
                html.Div([
                    html.Div("위험도 분류 기준", style={'fontSize': '14px', 'fontWeight': '700', 'marginBottom': '12px',
                                                 'color': '#2d3748'}),
                    *[html.Div([
                        html.Span(label,
                                  style={'background': bg, 'color': col, 'padding': '3px 10px', 'borderRadius': '20px',
                                         'fontSize': '12px', 'fontWeight': '700', 'marginRight': '8px'}),
                        html.Span(desc, style={'fontSize': '12px', 'color': '#4a5568'}),
                    ], style={'marginBottom': '8px'})
                        for label, bg, col, desc in [
                            ('고위험', '#fff5f5', '#e53e3e', '위기확률 70% 이상'),
                            ('중위험', '#fffbeb', '#d97706', '위기확률 60~70%'),
                        ]],
                ], style=section_s()),
            ], style={'flex': '1', 'marginLeft': '20px'}),
        ], style={'display': 'flex'}),
    ], style={'padding': '24px'})])


def render_analysis(selected, open_det, sel_cand, show_all=False):
    if not selected:
        return html.Div(html.Div([
            html.Div("", style={'fontSize': '48px', 'marginBottom': '16px'}),
            html.Div("왼쪽에서 학교를 선택해주세요", style={'fontSize': '16px', 'color': '#718096'}),
        ], style={'textAlign': 'center', 'paddingTop': '80px'}))

    uid_row = df_high[df_high['_uid'] == selected] if selected else pd.DataFrame()
    if uid_row.empty:
        uid_row = df_high[df_high['학교명'] == selected]
    if uid_row.empty: return html.Div("데이터 없음")
    row = uid_row.iloc[0]
    selected = row['학교명']
    prob = row['위기확률']
    color = risk_color(prob)

    content = [html.Div([

        html.Div([
            html.H3(selected, style={'fontWeight': '700', 'margin': '0 8px 0 0', 'color': '#1a202c',
                                     'display': 'inline'}),
            html.Span(
                row.get('군집명', ''),
                style={
                    'background': '#ebf8ff', 'color': '#2563eb',
                    'border': '1px solid #bee3f8',
                    'borderRadius': '12px', 'padding': '2px 10px',
                    'fontSize': '12px', 'fontWeight': '600',
                    'verticalAlign': 'middle'
                }
            ) if pd.notna(row.get('군집명')) else html.Span(),
        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '4px'}),
        html.Div(f"{row['지역']} · {row['시도교육청']}",
                 style={'fontSize': '13px', 'color': '#718096', 'marginBottom': '20px'}),

        html.Div([
            html.Div([
                html.Div("재학생 수", style={'fontSize': '12px', 'color': '#718096', 'marginBottom': '4px'}),
                html.Div(f"{int(row['학생수'])}명", style={'fontSize': '26px', 'fontWeight': '700'}),
                html.Div(" 감소 추세" if row['학생수_변화율'] < 0 else " 증가",
                         style={'fontSize': '11px', 'color': '#e53e3e' if row['학생수_변화율'] < 0 else '#38a169'}),
            ], style=card_s()),
            html.Div([
                html.Div("학급 수", style={'fontSize': '12px', 'color': '#718096', 'marginBottom': '4px'}),
                html.Div(f"{int(row['학급수'])}학급" if pd.notna(row.get('학급수')) else "-",
                         style={'fontSize': '26px', 'fontWeight': '700'}),
            ], style=card_s()),
            html.Div([
                html.Div("학령인구 감소율", style={'fontSize': '12px', 'color': '#718096', 'marginBottom': '4px'}),
                html.Div(f"{row['지역학령인구 변화율']:.1f}%",
                         style={'fontSize': '26px', 'fontWeight': '700',
                                'color': '#e53e3e' if row['지역학령인구 변화율'] < -3 else '#ed8936'}),
                html.Div("최근 5년 기준", style={'fontSize': '11px', 'color': '#a0aec0'}),
            ], style=card_s()),
            html.Div([
                html.Div("통폐합 위험도", style={'fontSize': '12px', 'color': '#718096', 'marginBottom': '4px'}),
                html.Div([
                    html.Span(f"{int(prob * 100)}", style={'fontSize': '36px', 'fontWeight': '700', 'color': color}),
                    html.Span("/100", style={'fontSize': '16px', 'color': '#a0aec0', 'marginLeft': '2px'}),
                    html.Span(f" — {risk_label(prob)}",
                              style={'fontSize': '13px', 'color': color, 'fontWeight': '700'}),
                ]),
                html.Div("학생수·예산·지역 인구 감소율 등 기반 모델 예측값",
                         style={'fontSize': '11px', 'color': '#a0aec0', 'marginTop': '4px', 'lineHeight': '1.4'}),
            ], style={**card_s(), 'borderLeft': f'4px solid {color}'}),
        ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr 1fr 1.5fr', 'gap': '12px',
                  'marginBottom': '20px'}),
        html.Div([
            html.Div("위험 주요 원인 (LIME 분석)", style={'fontSize': '14px', 'fontWeight': '700', 'color': '#2d3748',
                                                  'marginBottom': '10px', 'paddingBottom': '8px',
                                                  'borderBottom': '1px solid #e2e8f0'}),
            _lime_chart(selected),
        ], style={**section_s(), 'marginBottom': '20px'}),
    ], style={'padding': '24px'})]

    content.append(render_sim(selected, open_det, sel_cand, show_all))
    return html.Div(content)


def render_sim(selected, open_det, sel_cand, show_all=False):
    active = open_det or '통폐합'
    if active == '통합운영':
        detail = render_integ(selected, sel_cand)
    elif active == '유지':
        detail = render_maint(selected)
    else:
        detail = render_consol(selected, sel_cand, show_all)

    btn_style = lambda t: {
        'flex': '1', 'padding': '10px',
        'background': '#2563eb' if active == t else '#f7fafc',
        'color': 'white' if active == t else '#4a5568',
        'border': f'1px solid {"#2563eb" if active == t else "#e2e8f0"}',
        'borderRadius': '8px', 'fontSize': '14px', 'fontWeight': '700' if active == t else '400',
        'cursor': 'pointer', 'fontFamily': 'Noto Sans KR'
    }

    return html.Div([
        html.Hr(style={'borderColor': '#e2e8f0'}),
        html.Div([
            html.H4(f"{selected} 시뮬레이션 결과", style={'fontWeight': '700', 'marginBottom': '4px'}),
            html.P("통폐합 · 통합운영 · 유지 시나리오를 확인합니다.",
                   style={'fontSize': '13px', 'color': '#718096', 'marginBottom': '16px'}),
            html.Div([
                html.Button("통폐합 자세히 보기", id={'type': 'det-btn', 'index': '통폐합'}, n_clicks=0, style=btn_style('통폐합')),
                html.Button("통합운영 자세히 보기", id={'type': 'det-btn', 'index': '통합운영'}, n_clicks=0,
                            style=btn_style('통합운영')),
                html.Button("유지 자세히 보기", id={'type': 'det-btn', 'index': '유지'}, n_clicks=0, style=btn_style('유지')),
            ], style={'display': 'flex', 'gap': '12px', 'marginBottom': '20px'}),
        ], style={'padding': '0 24px 16px'}),
        detail,
    ])


def make_consol_map(row_high, data, sel_cand, show_all=False):
    HIGH_LAT = row_high['위도']
    HIGH_LON = row_high['경도']

    # sel_cand가 있으면 해당 후보학교 위치로 center 이동
    map_center = [HIGH_LAT, HIGH_LON]
    map_zoom = 11
    if sel_cand:
        for _, cand in data.iterrows():

            slat = cand.get('후보_위도')
            slon = cand.get('후보_경도')
            if pd.isna(slat) or pd.isna(slon):
                continue
            uid = f"{cand['후보학교명']}__{slat:.6f}_{slon:.6f}"
            if uid == sel_cand:
                map_center = [slat, slon]
                map_zoom = 13
                break

    markers = [dl.CircleMarker(center=[HIGH_LAT, HIGH_LON], radius=12, color='white', weight=2,
                               fillColor='#ef4444', fillOpacity=0.9,
                               children=dl.Tooltip(f"{row_high['학교명']} (고위험)"))]

    for _, cand in data.iterrows():
        rank = int(cand['후보순위'])

        # 토글 OFF면 5순위까지만 지도에 표시
        # 토글 ON이면 전체 후보 표시
        if not show_all and rank > 5:
            continue

        sname = cand['후보학교명']
        slat = cand.get('후보_위도')
        slon = cand.get('후보_경도')
        if pd.isna(slat) or pd.isna(slon):
            continue

        uid = f"{sname}__{slat:.6f}_{slon:.6f}"
        is_sel = (uid == sel_cand)
        fill = '#2563eb' if rank <= 3 else '#94a3b8'
        r = 11 if is_sel else 8

        markers.append(dl.CircleMarker(
            id={'type': 'cand-map', 'index': uid}, n_clicks=0,
            center=[slat, slon], radius=r, color='white', weight=2,
            fillColor=fill, fillOpacity=0.9,
            children=[
                dl.Tooltip(f"{rank}순위: {sname}"),
                dl.Popup(html.Div([
                    html.Div(f"{rank}순위", style={'fontWeight': '700', 'marginBottom': '4px',
                                                 'color': '#2563eb' if rank <= 3 else '#64748b'}),
                    html.Div(sname)
                ]))
            ]
        ))

    return dl.Map(
        children=[dl.TileLayer(url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png")] + markers,
        center=map_center, zoom=map_zoom,
        minZoom=MAP_MIN_ZOOM, maxBounds=KOREA_BOUNDS,
        style={'height': '420px', 'borderRadius': '10px', 'overflow': 'hidden'}
    )


def _no_data_panel(message):
    return html.Div([
        html.Div("📂", style={'fontSize': '32px', 'marginBottom': '8px', 'textAlign': 'center'}),
        html.Div(message, style={'color': '#a0aec0', 'fontSize': '14px', 'textAlign': 'center'}),
    ], style={'padding': '40px 0', 'margin': '0 24px 24px'})


def render_consol(selected, sel_cand, show_all=False):
    if df_consol.empty or '고위험학교명' not in df_consol.columns:
        return _no_data_panel("통폐합 시뮬레이션 결과 파일이 아직 없습니다.")
    data = df_consol[df_consol['고위험학교명'] == selected]
    rh = df_high[df_high['학교명'] == selected]
    if rh.empty: return _no_data_panel("학교 데이터를 찾을 수 없습니다.")
    row_high = rh.iloc[0]
    if data.empty:
        return html.Div([html.Div([
            html.Div("① 통폐합 시뮬레이션 상세", style={'fontSize': '15px', 'fontWeight': '700', 'marginBottom': '16px'}),
            html.Div("20km 이내 후보 학교가 없습니다.",
                     style={'color': '#a0aec0', 'fontSize': '14px', 'textAlign': 'center', 'padding': '40px 0'}),
        ], style={**section_s(), 'margin': '0 24px 24px'})])

    # 수정
    ranked = {c['후보학교명']: int(c['후보순위']) for _, c in data.sort_values('후보순위').iterrows()}
    sel_cand_name = sel_cand.split('__')[0] if sel_cand and '__' in sel_cand else sel_cand

    if not sel_cand:
        result_panel = html.Div("지도에서 학교를 클릭하거나 아래 순위 목록을 선택하면 결과가 표시됩니다.",
                                style={'color': '#a0aec0', 'fontSize': '13px', 'textAlign': 'center',
                                       'padding': '40px 0'})
    else:
        rank = ranked.get(sel_cand_name)
        sel_row = data[data['후보학교명'] == sel_cand_name]

        if not sel_row.empty:
            cand = sel_row.iloc[0]
            dist_chg = cand.get('통학거리_변화_km', None)
            time_chg = cand.get('통학시간_변화_분', None)
            if pd.notna(dist_chg):
                s = '+' if dist_chg >= 0 else ''
                dist_txt = f"{s}{dist_chg:.1f}km"
                dist_col = '#e53e3e' if dist_chg > 0 else '#38a169'
                time_txt = f"({'+' if time_chg >= 0 else ''}{time_chg:.0f}분)" if pd.notna(time_chg) else ''
            else:
                dist_txt = f"{cand.get('도로거리_km', '-')}km"
                dist_col = '#4a5568'
                time_txt = f"({cand['소요시간_분']}분)" if pd.notna(cand.get('소요시간_분')) else ''
            bus_c = '#38a169' if cand.get('버스접근성') == '양호' else '#e53e3e'

            def ir(label, value, sub=''):
                return html.Div([
                    html.Span(label, style={'color': '#718096', 'fontSize': '12px', 'width': '140px',
                                            'display': 'inline-block'}),
                    html.Span(value, style={'fontWeight': '700', 'fontSize': '13px'}),
                    html.Span(f"  {sub}", style={'color': '#a0aec0', 'fontSize': '11px'}) if sub else None,
                ], style={'marginBottom': '8px', 'display': 'flex', 'alignItems': 'center'})

            result_panel = html.Div([
                html.Div([
                    html.Span(f"{rank}순위 추천 후보" if rank else "후보 학교",
                              style={'background': '#2563eb', 'color': 'white', 'borderRadius': '4px',
                                     'padding': '2px 8px', 'fontSize': '11px', 'marginRight': '8px'}),
                    html.Span(sel_cand_name, style={'fontWeight': '700', 'fontSize': '16px'}),
                ], style={'marginBottom': '16px'}),
                ir("통학거리 변화",
                   html.Span(dist_txt, style={'color': dist_col, 'fontWeight': '700', 'fontSize': '13px'}), time_txt),
                ir("학급당 학생수",
                   f"{cand['통폐합후_학급당학생수']:.1f}명" if pd.notna(cand.get('통폐합후_학급당학생수')) else '-',
                   "(2025 초등학교 전국 평균 20명)"),
                ir("교원 1인당 학생수",
                   f"{cand['통폐합후_교원1인당']:.1f}명" if pd.notna(cand.get('통폐합후_교원1인당')) else '-',
                   "(OECD 기준 14.8명)"),
                ir("버스 접근성",
                   html.Span(cand.get('버스접근성', '-'), style={'color': bus_c, 'fontWeight': '700', 'fontSize': '13px'}),
                   f"({int(cand['버스정류장수'])}개 정류장)" if pd.notna(cand.get('버스정류장수')) else ''),
                ir("순 절감액",
                   f"{int(cand['순절감액_추정']):,}원" if pd.notna(cand.get('순절감액_추정')) else '-'),
                ir("스쿨존",
                   f"CCTV {int(cand['스쿨존_CCTV수'])}대  도로폭 {cand['스쿨존_도로폭']}m"
                   if pd.notna(cand.get('스쿨존_CCTV수')) else '-'),
                html.Div(f"종합점수 {cand['종합점수']:.3f}",
                         style={'fontSize': '12px', 'color': '#2563eb', 'fontWeight': '700',
                                'textAlign': 'right', 'marginTop': '8px'}),
            ], style={**section_s(), 'overflowY': 'auto', 'maxHeight': '420px'})
        else:
            result_panel = html.Div("추천 후보(1~3순위)만 상세 결과를 볼 수 있습니다.",
                                    style={'color': '#a0aec0', 'fontSize': '13px', 'textAlign': 'center',
                                           'padding': '40px 0'})

    # ── 후보 학교 순위 목록 (클릭 가능) ──────────────────
    rank_list = html.Div([
        html.Div("후보 학교 순위", style={'fontWeight': '700', 'fontSize': '14px',
                                    'marginBottom': '12px', 'marginTop': '20px'}),
        html.Div([
            html.Div(
                id={'type': 'rank-item', 'index': (
                    f"{c['후보학교명']}__{c['후보_위도']:.6f}_{c['후보_경도']:.6f}"
                    if pd.notna(c.get('후보_위도')) and pd.notna(c.get('후보_경도'))
                    else c['후보학교명']
                )},
                n_clicks=0,
                children=[
                    html.Span(f"{int(c['후보순위'])}",
                              style={
                                  'display': 'inline-flex', 'alignItems': 'center', 'justifyContent': 'center',
                                  'width': '24px', 'height': '24px', 'borderRadius': '50%',
                                  'background': '#2563eb' if int(c['후보순위']) <= 3 else '#94a3b8',
                                  'color': 'white', 'fontSize': '12px', 'fontWeight': '700', 'marginRight': '10px'
                              }),
                    html.Span(c['후보학교명'], style={'fontSize': '13px'}),
                ],
                style={
                    'display': 'flex', 'alignItems': 'center', 'padding': '8px 6px',
                    'borderBottom': '1px solid #edf2f7', 'cursor': 'pointer', 'borderRadius': '6px',
                    'background': '#ebf8ff' if (
                            (f"{c['후보학교명']}__{c['후보_위도']:.6f}_{c['후보_경도']:.6f}"
                             if pd.notna(c.get('후보_위도')) and pd.notna(c.get('후보_경도'))
                             else c['후보학교명']) == sel_cand
                    ) else 'transparent',
                }
            )
            for _, c in (
                data.sort_values('후보순위')
                if show_all
                else data.sort_values('후보순위').head(5)
            ).iterrows()
        ])
    ], style=section_s())

    return html.Div([html.Div([
        html.Div([
            html.Div("① 통폐합 시뮬레이션 상세",
                     style={'fontSize': '15px', 'fontWeight': '700'}),

            html.Div([
                html.Span("전체 후보 보기",
                          style={'fontSize': '12px', 'color': '#718096', 'marginRight': '8px'}),

                dcc.Checklist(
                    id='toggle-all-cands',
                    options=[{'label': '', 'value': 'show'}],
                    value=['show'] if show_all else [],
                    inputStyle={'cursor': 'pointer'},
                    style={'display': 'inline-block'}
                ),
            ], style={'display': 'flex', 'alignItems': 'center'}),
        ], style={
            'display': 'flex',
            'justifyContent': 'space-between',
            'alignItems': 'center',
            'marginBottom': '4px'
        }),

        html.Div(
            "🔵 파란 원 = 추천 후보(1~5순위) / 회색 원 = 그 외 후보  ※ 지도 클릭 또는 순위 목록 선택으로 상세 결과를 확인하세요",
            style={'fontSize': '12px', 'color': '#718096', 'marginBottom': '16px'}
        ),

        html.Div([
            html.Div([make_consol_map(row_high, data, sel_cand, show_all)],
                     style={'flex': '1.2'}),

            html.Div([
                result_panel,
                rank_list,
            ], style={'flex': '1', 'paddingLeft': '20px'}),
        ], style={'display': 'flex', 'gap': '16px'}),

    ], style={**section_s(), 'margin': '0 24px 24px'})])


def make_cand_map(row_high, cands, name_col, sel_cand):
    markers = [dl.CircleMarker(center=[row_high['위도'], row_high['경도']],
                               radius=12, color='white', weight=2,
                               fillColor='#ef4444', fillOpacity=0.9,
                               children=dl.Tooltip(f"{row_high['학교명']} (고위험)"))]
    colors = ['#3b82f6', '#22c55e', '#a855f7']
    lat_col = '통합후보_위도' if '통합후보_위도' in cands.columns else '후보_위도'
    lon_col = '통합후보_경도' if '통합후보_경도' in cands.columns else '후보_경도'

    # sel_cand 있으면 해당 위치로 center 이동
    map_center = [row_high['위도'], row_high['경도']]
    map_zoom = 11
    if sel_cand:
        for _, cand in cands.iterrows():
            if cand[name_col] == sel_cand:
                clat = cand.get(lat_col)
                clon = cand.get(lon_col)
                if pd.notna(clat) and pd.notna(clon):
                    map_center = [clat, clon]
                    map_zoom = 13
                break

    for i, (_, cand) in enumerate(cands.iterrows()):
        clat = cand.get(lat_col)
        clon = cand.get(lon_col)
        if pd.isna(clat) or pd.isna(clon):
            continue
        is_sel = (cand[name_col] == sel_cand)
        markers.append(dl.CircleMarker(
            id={'type': 'integ-map', 'index': cand[name_col]},
            n_clicks=0,
            center=[clat, clon], radius=14 if is_sel else 9,
            color='white', weight=2, fillColor=colors[i % 3], fillOpacity=0.9,
            children=dl.Tooltip(f"{i + 1}순위: {cand[name_col]}"),
        ))

    return dl.Map(
        children=[dl.TileLayer(url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png")] + markers,
        center=map_center, zoom=map_zoom,
        minZoom=MAP_MIN_ZOOM, maxBounds=KOREA_BOUNDS,
        style={'height': '360px', 'borderRadius': '10px', 'overflow': 'hidden'}
    )


def render_integ(selected, sel_cand):
    if df_integ.empty or '고위험학교명' not in df_integ.columns:
        return _no_data_panel("통합운영 시뮬레이션 결과 파일이 아직 없습니다.")
    data = df_integ[df_integ['고위험학교명'] == selected]
    rh = df_high[df_high['학교명'] == selected]
    if rh.empty:   return _no_data_panel("학교 데이터를 찾을 수 없습니다.")
    if data.empty: return _no_data_panel("3km 이내 통합 가능한 중학교가 없습니다.")
    row_high = rh.iloc[0]

    name_col = '통합후보_학교명' if '통합후보_학교명' in data.columns else '후보학교명'
    dist_col = '도로거리_km'
    time_col = '소요시간_분'

    cand_items = []
    for _, cand in data.sort_values('후보순위').head(5).iterrows():
        cname = cand[name_col]
        is_sel = (cname == sel_cand)
        dist_txt = f"{cand.get(dist_col, '-')}km"
        time_txt = f"({cand.get(time_col, '-')}분)" if pd.notna(cand.get(time_col)) else ''
        cost_sum = cand.get('운영비_합산', None)
        cost_txt = f"{int(cost_sum):,}원" if pd.notna(cost_sum) else '-'
        staff = cand.get('일반직_합산', None)
        staff_txt = f"일반직 합산 {int(staff)}명" if pd.notna(staff) else ''
        sg = cand.get('같은시군구', False)
        sigungu_txt = "✔ 같은 시군구" if (sg is True or sg == 'True' or sg == 1) else ''

        cand_items.append(
            html.Div(
                id=f'integ-wrap-{cname.replace(" ", "_")}',  # 스크롤용
                children=[
                    html.Div(
                        id={'type': 'cand', 'index': cname},
                        n_clicks=0,
                        children=[
                            html.Div(f"{int(cand.get('후보순위', 0))}순위: {cname} (중학교)",
                                     style={'fontWeight': '700', 'fontSize': '14px', 'marginBottom': '6px'}),
                            html.Div([
                                html.Span(f" {dist_txt} {time_txt}"), html.Br(),
                                html.Span(f" 운영비 합산: {cost_txt}"), html.Br(),
                                html.Span(staff_txt) if staff_txt else None,
                                html.Br() if staff_txt else None,
                                html.Span(sigungu_txt,
                                          style={'color': '#38a169', 'fontWeight': '600'}) if sigungu_txt else None,
                                html.Span(f"종합점수: {cand.get('종합점수', '-'):.3f}",
                                          style={'color': '#2563eb', 'fontSize': '11px'}) if pd.notna(
                                    cand.get('종합점수')) else None,
                            ], style={'fontSize': '13px', 'color': '#4a5568', 'lineHeight': '1.8'}),
                        ],
                        style={'background': '#ebf8ff' if is_sel else '#f7fafc', 'borderRadius': '10px',
                               'padding': '14px', 'marginBottom': '10px', 'cursor': 'pointer',
                               'border': f'2px solid {"#3b82f6" if is_sel else "transparent"}'}
                    )
                ]
            )
        )

    return html.Div([html.Div([
        html.Div("② 통합운영 시뮬레이션 상세", style={'fontSize': '15px', 'fontWeight': '700', 'marginBottom': '4px'}),
        html.Div("초등+중학교 통합운영 시 운영비·일반직 합산 기준 최적 후보",
                 style={'fontSize': '12px', 'color': '#718096', 'marginBottom': '16px'}),
        html.Div([
            html.Div([make_cand_map(row_high, data, name_col, sel_cand)], style={'flex': '1'}),
            html.Div([
                html.Div("통합 후보 중학교 목록", style={'fontWeight': '700', 'marginBottom': '12px', 'fontSize': '14px'}),
                html.Div(cand_items),
            ], style={'flex': '1', 'paddingLeft': '20px', 'overflowY': 'auto', 'maxHeight': '400px'}),
        ], style={'display': 'flex', 'gap': '16px'}),
    ], style={**section_s(), 'margin': '0 24px 24px'})])

def _run_prophet(school_code=None, school_name=None, sido_office=None, region=None, periods=10):
    if df_student_ts.empty:
        return pd.DataFrame(), pd.DataFrame()

    sub = pd.DataFrame()

    if school_code is not None and str(school_code).strip() not in ["", "nan", "None"]:
        sub = df_student_ts[
            df_student_ts["학교코드"].astype(str).str.strip() == str(school_code).strip()
        ].copy()

    if sub.empty and school_name is not None:
        sub = df_student_ts[
            df_student_ts["학교명"].astype(str).str.strip() == str(school_name).strip()
        ].copy()

        if not sub.empty and sido_office and "시도교육청" in sub.columns:
            sub2 = sub[
                sub["시도교육청"].astype(str).str.strip() == str(sido_office).strip()
            ].copy()
            sub = sub2

            # 시도교육청 필터 후에도 여러 연도에 중복 행이 있으면 지역으로 추가 필터링
            if not sub.empty and region and "지역" in sub.columns:
                sigungu = str(region).split()[-1]
                sub3 = sub[sub["지역"].astype(str).str.contains(sigungu, na=False)].copy()
                if not sub3.empty:
                    sub = sub3

    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()

    hist = (
        sub.groupby("연도", as_index=False)["학생수"]
        .median()
        .sort_values("연도")
    )

    train = pd.DataFrame({
        "ds": pd.to_datetime(hist["연도"].astype(int).astype(str) + "-01-01"),
        "y": hist["학생수"]
    })

    if len(train) < 3:
        return hist, pd.DataFrame()

    try:
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.01
        )

        model.fit(train)

        future = model.make_future_dataframe(periods=periods, freq="YS")
        forecast = model.predict(future)
        forecast["연도"] = forecast["ds"].dt.year

        for col in ["yhat", "yhat_upper", "yhat_lower"]:
            forecast[col] = forecast[col].clip(lower=0)

        return hist, forecast

    except Exception as e:
        print(f"[Prophet 오류] {school_name}: {e}")
        return hist, pd.DataFrame()


def _make_prophet_graph(school_code=None, school_name=None, sido_office=None, region=None):
    """
    실제 학생수 + Prophet 예측 그래프 생성
    """

    hist, forecast = _run_prophet(
        school_code=school_code,
        school_name=school_name,
        sido_office=sido_office,
        region=region,
        periods=7
    )

    if hist.empty:
        return html.Div(
            "해당 학교의 연도별 학생수 시계열 데이터가 없습니다.",
            style={'fontSize': '13px', 'color': '#a0aec0', 'textAlign': 'center', 'padding': '30px 0'}
        )

    if len(hist) < 3:
        return html.Div(
            f"학생수 데이터가 {len(hist)}년치만 존재하여 예측이 어렵습니다.",
            style={'fontSize': '13px', 'color': '#a0aec0', 'textAlign': 'center', 'padding': '30px 0'}
        )

    fig = go.Figure()

    # 실제 학생수
    fig.add_trace(go.Scatter(
        x=hist["연도"],
        y=hist["학생수"],
        mode="lines+markers",
        name="실제 학생수",
        line=dict(color="#2563eb", width=3),
        marker=dict(size=7)
    ))

    # 예측 학생수
    if not forecast.empty:
        last_year = int(hist["연도"].max())
        last_val = float(hist[hist["연도"] == last_year]["학생수"].values[0])
        future_part = forecast[forecast["연도"] >= last_year + 1].copy()

        # 연결용 포인트 추가
        connect = pd.DataFrame({
            "연도": [last_year],
            "yhat": [last_val],
            "yhat_upper": [last_val],
            "yhat_lower": [last_val]
        })
        future_part = pd.concat([connect, future_part]).reset_index(drop=True)

        fig.add_trace(go.Scatter(
            x=future_part["연도"],
            y=future_part["yhat"],
            mode="lines+markers",
            name="예측 학생수",
            line=dict(color="#ef4444", width=3, dash="dash"),
            marker=dict(size=6)
        ))

        fig.add_trace(go.Scatter(
            x=list(future_part["연도"]) + list(future_part["연도"])[::-1],
            y=list(future_part["yhat_upper"]) + list(future_part["yhat_lower"])[::-1],
            fill="toself",
            fillcolor="rgba(239, 68, 68, 0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="예측 범위"
        ))

    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        title=dict(
            text="연도별 학생수 추이 및 예측",
            font=dict(size=15)
        ),
        xaxis_title="연도",
        yaxis_title="학생수",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        font=dict(family="Noto Sans KR")
    )

    return dcc.Graph(
        figure=fig,
        config={"displayModeBar": False},
        style={"width": "100%"}
    )


def render_maint(selected):

    if df_maint.empty or '학교명' not in df_maint.columns:
        return _no_data_panel("분교유지 시뮬레이션 결과 파일이 아직 없습니다.")
    rows = df_maint[df_maint['학교명'] == selected]
    if rows.empty: return _no_data_panel("해당 학교의 유지 시뮬레이션 데이터가 없습니다.")
    m = rows.iloc[0]

    # Prophet 매칭용 정보
    school_code = m.get("학교코드", None)
    sido_office = m.get("시도교육청", None)
    region = m.get('지역', None)
    if region is None or pd.isna(region):
        high_match = df_high[df_high["학교명"] == selected]
        if not high_match.empty:
            region = high_match.iloc[0].get("지역", None)

    # df_maint에 학교코드/시도교육청이 없을 경우 df_high에서 보완
    if school_code is None or pd.isna(school_code):
        high_match = df_high[df_high["학교명"] == selected]
        if not high_match.empty:
            school_code = high_match.iloc[0].get("학교코드", None)

    if sido_office is None or pd.isna(sido_office):
        high_match = df_high[df_high["학교명"] == selected]
        if not high_match.empty:
            sido_office = high_match.iloc[0].get("시도교육청", None)

    cur_stu = m.get('현재_학생수', None)
    hist, forecast = _run_prophet(
        school_code=school_code,
        school_name=selected,
        sido_office=sido_office,
        region=region,
        periods=7
    )

    cur_cost = m.get('현재_연간운영비', None)
    cur_cost_per = m.get('현재_1인당운영비', None)

    if not forecast.empty:
        last_year = 2025
        stu_5yr = forecast[forecast['연도'] == last_year + 5]['yhat'].values
        stu_5yr = round(max(stu_5yr[0], 0), 1) if len(stu_5yr) > 0 else None

        stu_10yr = forecast[forecast['연도'] == last_year + 7]['yhat'].values
        stu_10yr = round(max(stu_10yr[0], 0), 1) if len(stu_10yr) > 0 else None

        eff_5yr = round(cur_cost / stu_5yr / cur_cost_per, 2) \
            if cur_cost and stu_5yr and stu_5yr > 0 and cur_cost_per else None
        eff_r = round(cur_cost / stu_10yr / cur_cost_per, 2) \
            if cur_cost and stu_10yr and stu_10yr > 0 and cur_cost_per else None
    else:
        stu_5yr = m.get('5년후_예측학생수', None)
        stu_10yr = m.get('7년후_예측학생수', None)
        eff_5yr = m.get('5년후_운영비효율배율', None)
        eff_r = m.get('7년후_운영비효율배율', None)
    rate = m.get('적용_평균변화율', None)
    eff_5c = '#e53e3e' if (pd.notna(eff_5yr) and eff_5yr > 1.5) else '#38a169'
    eff_c = '#e53e3e' if (pd.notna(eff_r) and eff_r > 1.5) else '#38a169'

    cost_yr_cols = sorted([c for c in m.index if str(c).endswith('년_연간운영비')])
    trend_items = []
    for col in cost_yr_cols:
        yr = str(col).replace('년_연간운영비', '')
        val = m[col]
        if pd.notna(val) and val > 0:
            trend_items.append(html.Div([
                html.Span(f"{yr}년",
                          style={'fontSize': '12px', 'color': '#718096', 'width': '60px', 'display': 'inline-block'}),
                html.Span(f"{int(val):,}원", style={'fontWeight': '700', 'fontSize': '13px'}),
            ], style={'marginBottom': '4px'}))

    # Prophet 학생수 예측 그래프 패널
    prophet_panel = html.Div([
        html.Div(
            "학생수 예측 그래프",
            style={
                'fontSize': '13px',
                'fontWeight': '700',
                'marginBottom': '10px',
                'color': '#2d3748'
            }
        ),
        _make_prophet_graph(
            school_code=school_code,
            school_name=selected,
            sido_office=sido_office,
            region=region
        )
    ], style={**section_s(), 'marginTop': '16px'})

    return html.Div([html.Div([
        html.Div("③ 분교유지 시뮬레이션 상세", style={'fontSize': '15px', 'fontWeight': '700', 'marginBottom': '16px'}),
        html.Div([
            html.Div([
                html.Div(lbl, style={'fontSize': '12px', 'color': '#718096', 'marginBottom': '4px'}),
                html.Div(f"{v:.0f}명" if pd.notna(v) else '-', style={'fontSize': '28px', 'fontWeight': '700'}),
            ], style={**card_s(), 'textAlign': 'center'})
            for lbl, v in [('현재', cur_stu), ('5년 후', stu_5yr), ('7년 후', stu_10yr)]
        ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr 1fr', 'gap': '12px', 'marginBottom': '16px'}),
        html.Div([
            html.Div([
                html.Div("적용 변화율", style={'fontSize': '12px', 'color': '#718096', 'marginBottom': '4px'}),
                html.Div(f"{rate:.2f}%/년" if pd.notna(rate) else '-', style={'fontSize': '20px', 'fontWeight': '700'}),
                html.Div("시군구 평균 학령인구 변화율", style={'fontSize': '11px', 'color': '#a0aec0', 'marginTop': '4px'}),
            ], style=card_s()),
            html.Div([
                html.Div("현재 1인당 운영비", style={'fontSize': '12px', 'color': '#718096', 'marginBottom': '4px'}),
                html.Div(f"{int(m['현재_1인당운영비']):,}원" if pd.notna(m.get('현재_1인당운영비')) else '-',
                         style={'fontSize': '20px', 'fontWeight': '700'}),
            ], style=card_s()),
            html.Div([
                html.Div("5년 후 운영비 효율", style={'fontSize': '12px', 'color': '#718096', 'marginBottom': '4px'}),
                html.Div(f"{eff_5yr:.2f}배" if pd.notna(eff_5yr) else '-',
                         style={'fontSize': '20px', 'fontWeight': '700', 'color': eff_5c}),
                html.Div("현재 대비 1인당 운영비 배율", style={'fontSize': '11px', 'color': '#a0aec0', 'marginTop': '4px'}),
            ], style=card_s()),
            html.Div([
                html.Div("7년 후 운영비 효율", style={'fontSize': '12px', 'color': '#718096', 'marginBottom': '4px'}),
                html.Div(f"{eff_r:.2f}배" if pd.notna(eff_r) else '-',
                         style={'fontSize': '20px', 'fontWeight': '700', 'color': eff_c}),
                html.Div("현재 대비 1인당 운영비 배율", style={'fontSize': '11px', 'color': '#a0aec0', 'marginTop': '4px'}),
            ], style=card_s()),
        ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr 1fr 1fr',
                  'gap': '12px', 'marginBottom': '16px' if trend_items else 0}),
        html.Div([
            html.Div("연도별 연간 운영비 추이",
                     style={'fontSize': '13px', 'fontWeight': '700', 'marginBottom': '10px', 'color': '#2d3748'}),
            *trend_items,
        ], style=section_s()) if trend_items else html.Div(),
        prophet_panel,
    ], style={**section_s(), 'margin': '0 24px 24px'})])


if __name__ == '__main__':
    app.run(debug=True, port=8050)

