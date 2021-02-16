from ta import *
import pandas as pd


def create_ta(df):
    """
    Creates a technical analysis dataframe with relevant OHLC features.
    
    Args:
        df (pandas.DataFrame): DataFrame from which to add data.
    
    Returns:
        Formatted DataFrame.
    """
    print("Loaded CSV...")
    df = add_all_ta_features(df, "open", "high", "low", "close", "volume", fillna=True) 
    print("Added TA Features...")
    return df

    
def clean_ta(df):
    """
    Cleans a dataframe by removing irrelevant columns.
    
    Args:
        df (pandas.DataFrame): DataFrame from which to delete columns.
    
    Returns:
        Formatted DataFrame.
    """
    df = df.reset_index(drop=True)
    del df['close']
    del df['ticker']
    del df['adjclose']
    del df['high']
    del df['low']
    del df['volatility_bbh']
    del df['volatility_atr']
    del df['volume_nvi']
    del df['volatility_bbl']
    del df['volatility_bbhi']
    del df['volatility_bbli']
    del df['volatility_bbm']
    del df['volatility_kcc']
    del df['volatility_kch']
    del df['volatility_kcl']
    del df['volatility_dch']
    del df['volatility_dcl']
    del df['volume_em']
    del df['trend_kst']
    del df['trend_kst_diff']
    del df['trend_ema_fast']
    del df['trend_ema_slow']
    del df['trend_ichimoku_a']
    del df['trend_ichimoku_b']
    del df['trend_visual_ichimoku_b']
    del df['trend_visual_ichimoku_a']
    # del df['momentum_kama']
    del df['momentum_wr']
    del df['momentum_stoch_signal']
    del df['momentum_stoch']
    del df['momentum_uo']
    del df['momentum_rsi']
    del df['trend_macd']
    del df['trend_macd_signal']
    del df['trend_adx_pos']
    del df['trend_adx_neg']
    del df['trend_vortex_diff']
    del df['trend_cci']
    del df['trend_kst_sig']
    del df['trend_aroon_up']
    del df['trend_aroon_ind']
    del df['others_dlr']
    del df['others_cr']
    del df['others_dr']
    return df








