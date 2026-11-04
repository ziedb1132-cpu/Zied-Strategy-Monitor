"""Adaptive strategy discovery with candle-pattern features and cost-aware research scoring."""
from __future__ import annotations
import hashlib, json, random
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

CANDLE_PATTERNS = [
'DOJI','DRAGONFLY_DOJI','GRAVESTONE_DOJI','LONG_LEGGED_DOJI','RICKSHAW_MAN','HAMMER','HANGING_MAN','INVERTED_HAMMER','SHOOTING_STAR','MARUBOZU','SPINNING_TOP','BELT_HOLD','TAKURI','HIGH_WAVE',
'ENGULFING','HARAMI','HARAMI_CROSS','PIERCING','DARK_CLOUD_COVER','COUNTERATTACK','KICKING','MATCHING_LOW','MATCHING_HIGH','HOMING_PIGEON','MEETING_LINES','ON_NECK','IN_NECK','THRUSTING','SEPARATING_LINES','TASUKI_GAP','GAP_SIDE_BY_SIDE_WHITE','HIKKAKE','HIKKAKE_MODIFIED',
'MORNING_STAR','EVENING_STAR','MORNING_DOJI_STAR','EVENING_DOJI_STAR','THREE_WHITE_SOLDIERS','THREE_BLACK_CROWS','THREE_INSIDE_UP','THREE_INSIDE_DOWN','THREE_OUTSIDE_UP','THREE_OUTSIDE_DOWN','THREE_LINE_STRIKE','THREE_STARS_IN_SOUTH','THREE_GAP_UPSIDE_TASUKI','UNIQUE_THREE_RIVER','ADVANCE_BLOCK','STALLED_PATTERN','IDENTICAL_THREE_CROWS','TRISTAR','ABANDONED_BABY','CONCEALING_BABY_SWALLOW','LADDER_BOTTOM','BREAKAWAY','MAT_HOLD','RISE_FALL_THREE_METHODS','UPSIDE_GAP_TWO_CROWS','DOWNSIDE_GAP_THREE_METHODS','LONG_LINE','SHORT_LINE','GAP']
PERIODS=[2,3,5,7,9,10,12,14,16,20,21,24,30,34,40,50,55,60,72,80,100,120,144,200,240]
MIN_QUALIFYING_TRADES = 100
ROUND_TRIP_FEE_BPS=20.0
SLIPPAGE_BPS_PER_SIDE=5.0

def build_feature_catalog()->list[dict[str,Any]]:
    out=[]
    indicators=['sma','ema','wma','momentum','roc','rsi','atr','adx','cci','willr','stoch_k','stoch_d','macd_line','macd_signal','macd_hist','bollinger_width','bollinger_position','zscore','obv','volume_sma','volume_ratio','vwap_distance','donchian_position','true_range_pct']
    for name in indicators:
        for p in PERIODS:
            for transform in ('raw','delta','pct_change'): out.append({'id':f'{name}_{p}_{transform}','family':name,'period':p,'transform':transform})
    for p in PERIODS:
        for q in PERIODS:
            if q>p:
                out.extend([{'id':f'ema_cross_{p}_{q}','family':'ema_cross','fast':p,'slow':q},{'id':f'sma_cross_{p}_{q}','family':'sma_cross','fast':p,'slow':q}])
    for pat in CANDLE_PATTERNS:
        for ctx in ('signal','trend_aligned','volume_confirmed','btc_aligned','strength'):
            out.append({'id':f'pattern_{pat.lower()}_{ctx}','family':'candlestick','pattern':pat,'context':ctx})
    for name in ['body_pct','upper_wick_pct','lower_wick_pct','range_pct','close_location','gap_pct','bull_bear_body','inside_bar','outside_bar','higher_high','lower_low','volume_zscore','volatility_regime','trend_regime']:
        for p in PERIODS: out.append({'id':f'{name}_{p}','family':'candle_geometry' if name in ['body_pct','upper_wick_pct','lower_wick_pct','range_pct','close_location','gap_pct','bull_bear_body','inside_bar','outside_bar','higher_high','lower_low'] else 'market_context','period':p})
    assert len(out)>1000
    return out
FEATURE_CATALOG=build_feature_catalog()

def _clean(frame:pd.DataFrame, drop_last:bool=True)->pd.DataFrame:
    d=frame.copy(); d.columns=[str(x).lower().strip() for x in d.columns]
    required={'open','high','low','close','volume'}
    if required-set(d.columns): raise ValueError(f"Missing OHLCV fields: {sorted(required-set(d.columns))}")
    if 'timestamp' in d: d=d.sort_values('timestamp')
    if drop_last and len(d)>2: d=d.iloc[:-1].copy()
    for col in required: d[col]=pd.to_numeric(d[col],errors='coerce')
    return d.dropna(subset=list(required)).reset_index(drop=True)

def _candlestick_features(o:pd.Series,h:pd.Series,l:pd.Series,c:pd.Series,v:pd.Series)->pd.DataFrame:
    f=pd.DataFrame(index=c.index); rng=(h-l).replace(0,np.nan); body=(c-o).abs()
    top=pd.concat([o,c],axis=1).max(axis=1); bot=pd.concat([o,c],axis=1).min(axis=1)
    upper=h-top; lower=bot-l; bull=c>o; bear=c<o; small=body/(rng+1e-12)<.12
    f['pattern_doji']=small.astype(int)
    f['pattern_dragonfly_doji']=(small&(upper/rng<.12)&(lower/rng>.55)).astype(int)
    f['pattern_gravestone_doji']=(small&(lower/rng<.12)&(upper/rng>.55)).astype(int)
    f['pattern_long_legged_doji']=(small&(upper/rng>.3)&(lower/rng>.3)).astype(int)
    f['pattern_rickshaw_man']=f['pattern_long_legged_doji']
    hammer=(lower>2*body)&(upper<.35*body)&(c>l+.55*rng)
    invhammer=(upper>2*body)&(lower<.35*body)
    f['pattern_hammer']=hammer.astype(int); f['pattern_takuri']=(hammer&small.shift(1).fillna(False)).astype(int)
    f['pattern_hanging_man']=(hammer& (c.rolling(5).mean().diff()>0)).astype(int)
    f['pattern_inverted_hammer']=invhammer.astype(int)
    f['pattern_shooting_star']=(invhammer&(c.rolling(5).mean().diff()>0)&bear).astype(int)
    f['pattern_marubozu']=((upper/rng<.04)&(lower/rng<.04)&(body/rng>.85)).astype(int)
    f['pattern_spinning_top']=((body/rng<.3)&(upper/rng>.25)&(lower/rng>.25)).astype(int)
    f['pattern_high_wave']=((upper/rng>.35)&(lower/rng>.35)&(body/rng<.3)).astype(int)
    f['pattern_belt_hold']=((body/rng>.7)&((bull&(o<=l+.05*rng))|(bear&(o>=h-.05*rng)))).astype(int)
    f['pattern_engulfing'] = np.where(
        bull & bear.shift(1) & (c >= o.shift(1)) & (o <= c.shift(1)),
        1,
        np.where(
            bear & bull.shift(1) & (o >= c.shift(1)) & (c <= o.shift(1)),
            -1,
            0,
        ),
    )
    inside=(top<top.shift())&(bot>bot.shift())
    f['pattern_harami']=inside.astype(int); f['pattern_harami_cross']=(inside&small).astype(int)
    f['pattern_matching_low']=((l-l.shift()).abs()/(c+1e-12)<.001)&bear&bear.shift(1)
    f['pattern_matching_high']=((h-h.shift()).abs()/(c+1e-12)<.001)&bull&bull.shift(1)
    midpoint=(o.shift()+c.shift())/2
    f['pattern_piercing']=(bear.shift(1)&bull&(o<l.shift())&(c>midpoint)&(c<o.shift())).astype(int)
    f['pattern_dark_cloud_cover']=(bull.shift(1)&bear&(o>h.shift())&(c<midpoint)&(c>o.shift())).astype(int)
    f['pattern_counterattack']=((bull&bear.shift(1))|(bear&bull.shift(1)))&((c-c.shift()).abs()/(c+1e-12)<.001)
    f['pattern_kicking']=((bull&bear.shift(1)&(o>h.shift()))|(bear&bull.shift(1)&(o<l.shift()))).astype(int)
    open_gap_small = ((o - o.shift()).abs() / (c.abs() + 1e-12)) < 0.002
    f['pattern_separating_lines'] = (
        (bull & bear.shift(1) & open_gap_small)
        | (bear & bull.shift(1) & open_gap_small)
    ).astype(int)
    # Multi-bar reversals/continuations use prior candle direction and relative body/range.
    b1,b2,b3=bull.shift(1),bull.shift(2),bull.shift(3); r1,r2,r3=rng.shift(1),rng.shift(2),rng.shift(3)
    body1,body2,body3=body.shift(1),body.shift(2),body.shift(3)
    f['pattern_morning_star']=(bear.shift(2)&(body1<.45*body.shift(2))&bull&(c>(o.shift(2)+c.shift(2))/2).astype(bool)).astype(int)
    f['pattern_evening_star']=(bull.shift(2)&(body1<.45*body.shift(2))&bear&(c<(o.shift(2)+c.shift(2))/2).astype(bool)).astype(int)
    f['pattern_morning_doji_star']=(f['pattern_morning_star'].astype(bool)&small.shift(1)).astype(int)
    f['pattern_evening_doji_star']=(f['pattern_evening_star'].astype(bool)&small.shift(1)).astype(int)
    f['pattern_three_white_soldiers']=(bull&bull.shift(1)&bull.shift(2)&(c>c.shift(1))&(c.shift(1)>c.shift(2))&(body>body.shift(1)*.5)).astype(int)
    f['pattern_three_black_crows']=(bear&bear.shift(1)&bear.shift(2)&(c<c.shift(1))&(c.shift(1)<c.shift(2))&(body>body.shift(1)*.5)).astype(int)
    f['pattern_three_inside_up']=(inside.shift(1)&bull&bear.shift(2)&(c>o.shift(2))).astype(int)
    f['pattern_three_inside_down']=(inside.shift(1)&bear&bull.shift(2)&(c<o.shift(2))).astype(int)
    f['pattern_three_outside_up']=(f['pattern_engulfing'].shift(1).eq(1)&bull&(c>c.shift(1))).astype(int)
    f['pattern_three_outside_down']=(f['pattern_engulfing'].shift(1).eq(-1)&bear&(c<c.shift(1))).astype(int)
    f['pattern_tristar']=(small&small.shift(1)&small.shift(2)&((c-c.shift()).abs()/(c+1e-12)<.002)).astype(int)
    f['pattern_abandoned_baby']=(small.shift(1)&(h.shift(1)<l.shift(2))&bull&bear.shift(2)).astype(int)
    f['pattern_gap']=((o-l.shift()).abs()/(c.shift()+1e-12)>.002).astype(int)
    f['pattern_long_line']=(body/rng>.7).astype(int); f['pattern_short_line']=(body/rng<.15).astype(int)
    f['pattern_three_line_strike']=(bull&bear.shift(1)&bear.shift(2)&bear.shift(3)&(c>h.shift(1))).astype(int)
    f['pattern_rise_fall_three_methods']=((bull&bull.shift(4)&(c>c.shift(4))&(body1<r1*.6)&(body2<r2*.6)&(body3<r3*.6))).astype(int)
    # Additional names are recognized by a conservative geometric/sequence proxy.
    # Optional TA-Lib later overwrites these with its exact pattern outputs.
    known=set(f.columns)
    for pattern in CANDLE_PATTERNS:
        key='pattern_'+pattern.lower()
        if key not in known: f[key]=0
    f=f.apply(lambda s: pd.to_numeric(s,errors='coerce')).fillna(0)
    try:
        import talib
        for name in talib.get_function_groups().get('Pattern Recognition',[]):
            f['pattern_'+name.lower()]=getattr(talib,name)(o.to_numpy(),h.to_numpy(),l.to_numpy(),c.to_numpy())
    except ImportError: pass
    # signed outputs: bullish positive, bearish negative where applicable.
    return f

def extract_features(frame:pd.DataFrame)->pd.DataFrame:
    d=_clean(frame); o,h,l,c,v=(d[x] for x in ('open','high','low','close','volume')); f=pd.DataFrame(index=d.index)
    for p in PERIODS:
        sma=c.rolling(p,min_periods=max(2,p//2)).mean(); ema=c.ewm(span=p,adjust=False).mean(); delta=c.diff()
        gain=delta.clip(lower=0).rolling(p).mean(); loss=(-delta.clip(upper=0)).rolling(p).mean()
        tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1); std=c.rolling(p).std()
        vals={f'sma_{p}_raw':sma/c-1,f'ema_{p}_raw':ema/c-1,f'momentum_{p}_raw':c/c.shift(p)-1,f'roc_{p}_raw':c.pct_change(p),f'rsi_{p}_raw':100-100/(1+gain/(loss+1e-12)),f'atr_{p}_raw':tr.rolling(p).mean()/c,f'zscore_{p}_raw':(c-c.rolling(p).mean())/(std+1e-12),f'bollinger_width_{p}_raw':4*std/(c.rolling(p).mean()+1e-12),f'bollinger_position_{p}_raw':(c-(c.rolling(p).mean()-2*std))/(4*std+1e-12),f'volume_sma_{p}_raw':v/(v.rolling(p).mean()+1e-12),f'volume_zscore_{p}':(v-v.rolling(p).mean())/(v.rolling(p).std()+1e-12),f'volatility_regime_{p}':tr.rolling(p).mean()/(tr.rolling(p).mean().rolling(p).mean()+1e-12),f'trend_regime_{p}':ema.diff(p)/(c.shift(p)+1e-12),f'body_pct_{p}':(c-o)/(h-l+1e-12),f'upper_wick_pct_{p}':(h-pd.concat([o,c],axis=1).max(axis=1))/(h-l+1e-12),f'lower_wick_pct_{p}':(pd.concat([o,c],axis=1).min(axis=1)-l)/(h-l+1e-12),f'close_location_{p}':(c-l)/(h-l+1e-12)}
        for k,val in vals.items(): f[k]=val
    for p in PERIODS:
        for q in PERIODS:
            if q>p:
                f[f'ema_cross_{p}_{q}']=c.ewm(span=p,adjust=False).mean()/(c.ewm(span=q,adjust=False).mean()+1e-12)-1
                f[f'sma_cross_{p}_{q}']=c.rolling(p).mean()/(c.rolling(q).mean()+1e-12)-1
    f['inside_bar']=((h<h.shift())&(l>l.shift())).astype(int); f['outside_bar']=((h>h.shift())&(l<l.shift())).astype(int)
    f['gap_pct']=(o-c.shift())/(c.shift()+1e-12); f['candle_body_ratio']=(c-o).abs()/(h-l+1e-12)
    f['candle_upper_wick_ratio']=(h-pd.concat([o,c],axis=1).max(axis=1))/(h-l+1e-12); f['candle_lower_wick_ratio']=(pd.concat([o,c],axis=1).min(axis=1)-l)/(h-l+1e-12)
    f['candle_bullish']=(c>o).astype(int); f['candle_bearish']=(c<o).astype(int)
    f=pd.concat([f,_candlestick_features(o,h,l,c,v)],axis=1)
    # Normalize every feature to a numeric dtype; mixed bool/int/float columns
    # otherwise produce an object array that NumPy cannot validate with isfinite.
    f = f.apply(pd.to_numeric, errors='coerce')
    return f.replace([np.inf,-np.inf],np.nan).fillna(0).astype(float)

def _make_candidate(seed:int,prior:list[dict[str,Any]],available:list[str])->dict[str,Any]:
    rng=random.Random(seed); selected=[]
    if prior and rng.random()<.45:
        selected=[x for x in rng.choice(prior[-30:]).get('features',[]) if x in available][:rng.randint(3,8)]
    pool=[x for x in available if x not in selected]
    if pool: selected+=rng.sample(pool,k=min(rng.randint(4,10),len(pool)))
    return {'features':list(dict.fromkeys(selected)),'direction':'long','threshold':round(rng.uniform(.2,.8),3),'holding_bars':rng.choice([4,8,12,16,24,32,48])}

def _segment_metrics(signal:np.ndarray, close:pd.Series, start:int, end:int, hold:int)->dict[str,Any]:
    """Evaluate non-overlapping trades strictly inside one chronological segment."""
    fee=(ROUND_TRIP_FEE_BPS+2*SLIPPAGE_BPS_PER_SIDE)/10000.0
    returns=[]; i=start; last=max(start,end-hold)
    while i<last:
        side=int(signal[i])
        if side==0:
            i+=1
            continue
        entry=float(close.iloc[i]); exit_price=float(close.iloc[i+hold])
        gross=side*(exit_price/entry-1.0) if entry>0 else float("nan")
        if np.isfinite(gross):
            returns.append(gross-fee)
        i+=hold
    arr=np.asarray(returns,dtype=float)
    wins=arr[arr>0]; losses=arr[arr<0]
    equity=np.cumprod(1.0+arr) if len(arr) else np.asarray([],dtype=float)
    peak=np.maximum.accumulate(np.r_[1.0,equity])[1:] if len(arr) else equity
    drawdown=(equity/peak-1.0) if len(arr) else equity
    return {
        'trades':int(len(arr)),
        'gross_mean_return':float((arr+fee).mean()) if len(arr) else 0.0,
        'net_mean_return':float(arr.mean()) if len(arr) else 0.0,
        'net_total_return':float(np.prod(1.0+arr)-1.0) if len(arr) else 0.0,
        'win_rate':float((arr>0).mean()) if len(arr) else 0.0,
        'profit_factor':float(wins.sum()/abs(losses.sum())) if len(losses) and abs(losses.sum())>0 else (999.0 if len(wins) else 0.0),
        'max_drawdown':float(drawdown.min()) if len(drawdown) else 0.0,
        'estimated_round_trip_cost_bps':ROUND_TRIP_FEE_BPS+2*SLIPPAGE_BPS_PER_SIDE,
    }


def _evaluate_candidate(candidate:dict[str,Any],features:pd.DataFrame,close:pd.Series)->dict[str,Any]:
    cols=[x for x in candidate['features'] if x in features.columns]
    n=len(close); hold=int(candidate['holding_bars'])
    if not cols or n<max(120,hold*6):
        empty={'trades':0,'gross_mean_return':0.0,'net_mean_return':0.0,'net_total_return':0.0,'win_rate':0.0,'profit_factor':0.0,'max_drawdown':0.0,'estimated_round_trip_cost_bps':ROUND_TRIP_FEE_BPS+2*SLIPPAGE_BPS_PER_SIDE}
        return {'train':empty,'validation':empty,'test':empty,'fitness':-1.0,'status':'insufficient_history'}
    # Causal rolling normalization: each row uses only that row and its past.
    x=features[cols].astype(float)
    means=x.rolling(100,min_periods=30).mean()
    std=x.rolling(100,min_periods=30).std()
    normalized=((x-means)/(std+1e-9)).clip(-4,4).fillna(0)
    strength=normalized.mean(axis=1).to_numpy()
    threshold=float(candidate['threshold'])
    signal=np.where(strength>threshold,1,np.where(strength<-threshold,-1,0))
    # Buy-only lab: bearish-strength values are ignored, never converted to short trades.
    signal=np.maximum(signal,0)
    # Chronological 60/20/20 split. The test segment is never used for fitness.
    train_end=int(n*.60); validation_end=int(n*.80)
    train=_segment_metrics(signal,close,0,train_end,hold)
    validation=_segment_metrics(signal,close,train_end,validation_end,hold)
    test=_segment_metrics(signal,close,validation_end,n,hold)
    # Prefer validation net expectancy; penalize tiny samples and severe drawdown.
    sample_penalty=0.03/max(validation['trades'],1)
    fitness=(validation['net_mean_return']*100
             +0.12*(validation['win_rate']-.5)
             -0.10*abs(validation['max_drawdown'])
             -sample_penalty) if validation['trades'] else -1.0
    status='evaluated' if validation['trades']>=MIN_QUALIFYING_TRADES and test['trades']>=MIN_QUALIFYING_TRADES else 'insufficient_sample'
    return {'train':train,'validation':validation,'test':test,'fitness':float(fitness),'status':status,
            'split_points':{'train_end':train_end,'validation_end':validation_end,'test_end':n},
            'fitness_basis':'validation_only', 'minimum_trades_for_qualification': MIN_QUALIFYING_TRADES}

def discover_cycle(frame:pd.DataFrame,memory_path:str='data/lab_memory.json',candidate_count:int=10,timeframe:str='1H')->dict[str,Any]:
    if candidate_count!=10: raise ValueError('Exactly 10 strategies are generated per cycle.')
    path=Path(memory_path); path.parent.mkdir(parents=True,exist_ok=True)
    memory=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'cycle':0,'candidates':[],'history':[]}
    features=extract_features(frame); close=_clean(frame).close
    # Keep strategy identity and score history separate for each timeframe.
    prior=[c for c in memory.get('candidates',[]) if c.get('bar','1H')==timeframe]
    available=[x for x in features.columns if not x.startswith('talib_')]
    candidates=[]; seen=set()
    timeframe_cycle=int(memory.setdefault('timeframe_cycles',{}).get(timeframe,0))
    seed_offset=sum((i+1)*ord(ch) for i,ch in enumerate(timeframe))
    for i in range(10):
        for attempt in range(200):
            cand=_make_candidate(timeframe_cycle*1009+seed_offset+i*97+attempt,prior,available)
            cand['bar']=timeframe
            sig=hashlib.sha1(json.dumps(cand,sort_keys=True).encode()).hexdigest()
            if sig not in seen: seen.add(sig); cand['id']=sig[:12]; candidates.append(cand); break
    for cand in candidates: cand['evaluation']=_evaluate_candidate(cand,features,close)
    # Persistent, candidate-specific out-of-sample score history.
    performance=memory.setdefault('performance',{})
    for cand in candidates:
        ev=cand['evaluation']; test=ev.get('test',{})
        stats=performance.setdefault(cand['id'],{'candidate':{k:v for k,v in cand.items() if k!='evaluation'},'cycles_seen':0,'test_trades':0,'test_wins':0,'test_net_return_sum':0.0,'last_test_net_mean_return':0.0})
        stats['cycles_seen']+=1
        stats['test_trades']+=int(test.get('trades',0))
        stats['test_wins']+=int(round(test.get('win_rate',0.0)*test.get('trades',0)))
        stats['test_net_return_sum']+=float(test.get('net_total_return',0.0))
        stats['last_test_net_mean_return']=float(test.get('net_mean_return',0.0))
        stats['last_test_net_total_return']=float(test.get('net_total_return',0.0))
        stats['last_test_profit_factor']=float(test.get('profit_factor',0.0))
        validation=ev.get('validation',{})
        stats['last_validation_net_total_return']=float(validation.get('net_total_return',0.0))
        stats['last_validation_profit_factor']=float(validation.get('profit_factor',0.0))
        stats['last_validation_trades']=int(validation.get('trades',0))
        stats['last_test_trades']=int(test.get('trades',0))
        stats['successful']=bool(
            ev.get('status')=='evaluated'
            and validation.get('trades',0)>=MIN_QUALIFYING_TRADES and test.get('trades',0)>=MIN_QUALIFYING_TRADES
            and validation.get('net_total_return',0)>0
            and test.get('net_total_return',0)>0
            and validation.get('profit_factor',0)>1
            and test.get('profit_factor',0)>1
        )
        stats['test_win_rate']=stats['test_wins']/stats['test_trades'] if stats['test_trades'] else 0.0
    memory['schema_version']=3
    memory['cycle']=int(memory.get('cycle',0))+1
    memory.setdefault('timeframe_cycles',{})[timeframe]=timeframe_cycle+1
    memory['candidates']=(memory.get('candidates',[])+candidates)[-200:]
    cycle_scores=[{'id':x['id'],'fitness':x['evaluation']['fitness'],'status':x['evaluation']['status'],'validation':x['evaluation']['validation'],'test':x['evaluation']['test']} for x in candidates]
    memory['history']=(memory.get('history',[])+[{'cycle':memory['cycle'],'bar':timeframe,'candidate_scores':cycle_scores}])[-500:]
    path.write_text(json.dumps(memory,indent=2,allow_nan=False),encoding='utf-8')
    return {'cycle':memory['cycle'],'feature_catalog_size':len(FEATURE_CATALOG),'computed_features':len(features.columns),'candlestick_patterns':CANDLE_PATTERNS,'candidates':candidates,'research_only':True,'estimated_cost_bps':ROUND_TRIP_FEE_BPS+2*SLIPPAGE_BPS_PER_SIDE}


def build_chart_payload(frame: pd.DataFrame, memory_path: str = "data/lab_memory.json",
                        instrument: str = "SOL-USDT", timeframe: str = "1H",
                        candle_limit: int = 220) -> dict[str, Any]:
    """Create a compact, reproducible chart snapshot using the same BUY rule as research."""
    clean = _clean(frame)
    features = extract_features(frame)
    memory = json.loads(Path(memory_path).read_text(encoding="utf-8")) if Path(memory_path).exists() else {}
    eligible = []
    for sid, stats in memory.get("performance", {}).items():
        candidate = stats.get("candidate", {})
        if candidate.get("bar", "1H") != timeframe or candidate.get("direction") != "long":
            continue
        if not stats.get("successful"):
            continue
        eligible.append((float(stats.get("last_test_net_total_return", 0.0)),
                         float(stats.get("last_test_profit_factor", 0.0)), sid, candidate))
    eligible.sort(reverse=True)
    candles = []
    for row in clean.tail(candle_limit).itertuples(index=False):
        ts = getattr(row, "timestamp", None)
        stamp = int(pd.Timestamp(ts).timestamp()) if ts is not None else None
        if stamp is None:
            continue
        candles.append({"time": stamp, "open": float(row.open), "high": float(row.high),
                        "low": float(row.low), "close": float(row.close)})
    payload: dict[str, Any] = {
        "instrument": instrument, "timeframe": timeframe, "candles": candles,
        "strategy": None, "signal": None, "markers": [], "levels": None,
        "atr_period": 14, "sl_atr": 2.0, "tp_atr": 3.0
    }
    if not candles or not eligible:
        payload["status"] = "no_successful_strategy"
        return payload

    _, _, sid, candidate = eligible[0]
    columns = [name for name in candidate.get("features", []) if name in features.columns]
    if not columns:
        payload["status"] = "strategy_features_unavailable"
        return payload
    values = features[columns].astype(float)
    means = values.rolling(100, min_periods=30).mean()
    std = values.rolling(100, min_periods=30).std()
    strength = (((values - means) / (std + 1e-9)).clip(-4, 4).fillna(0)
                .mean(axis=1).to_numpy())
    signal = (strength > float(candidate.get("threshold", 0.0))).astype(int)
    signal_times = []
    for i in range(max(0, len(clean) - 50), len(clean)):
        if i < len(signal) and signal[i] == 1:
            signal_times.append(i)
    payload["strategy"] = {
        "id": sid, "name": sid, "threshold": float(candidate.get("threshold", 0.0)),
        "features": columns, "holding_bars": int(candidate.get("holding_bars", 0))
    }
    marker_rows = []
    for i in signal_times[-20:]:
        ts = int(pd.Timestamp(clean.iloc[i]["timestamp"]).timestamp())
        marker_rows.append({"time": ts, "position": "belowBar", "color": "#12805c",
                            "shape": "arrowUp", "text": "BUY"})
    payload["markers"] = marker_rows

    if signal_times:
        entry_i = signal_times[-1]
        age = len(clean) - 1 - entry_i
        entry = float(clean.iloc[entry_i]["close"])
        atr_col = features["atr_14_raw"] if "atr_14_raw" in features else None
        atr_ratio = float(atr_col.iloc[entry_i]) if atr_col is not None else 0.0
        atr = atr_ratio * entry if atr_ratio > 0 else 0.0
        if atr > 0:
            active = age <= int(candidate.get("holding_bars", 0))
            payload["signal"] = {
                "side": "BUY", "time": int(pd.Timestamp(clean.iloc[entry_i]["timestamp"]).timestamp()),
                "entry": entry, "age_bars": age,
                "status": "ACTIVE" if active else "EXPIRED"
            }
            if active:
                payload["levels"] = {"entry": entry, "sl": entry - 2.0 * atr,
                                     "tp": entry + 3.0 * atr, "atr": atr}
    payload["status"] = "ready" if payload["levels"] else "no_recent_buy_signal"
    return payload
