import pandas as pd
from src.tennis_predictions import predict_tennis_match


def test_tennis_matchup_prediction_returns_probabilities():
    matches = pd.DataFrame([
        {"tourney_date":"20260101","winner_name":"A","loser_name":"B","surface":"Hard","w_svpt":70,"w_1stWon":45,"w_2ndWon":15,"w_ace":8,"w_df":2,"l_svpt":65,"l_1stWon":35,"l_2ndWon":10,"l_ace":5,"l_df":3},
        {"tourney_date":"20260105","winner_name":"A","loser_name":"C","surface":"Hard","w_svpt":60,"w_1stWon":38,"w_2ndWon":12,"w_ace":6,"w_df":1,"l_svpt":58,"l_1stWon":30,"l_2ndWon":11,"l_ace":4,"l_df":2},
        {"tourney_date":"20260110","winner_name":"B","loser_name":"C","surface":"Hard","w_svpt":62,"w_1stWon":40,"w_2ndWon":11,"w_ace":7,"w_df":2,"l_svpt":61,"l_1stWon":32,"l_2ndWon":12,"l_ace":4,"l_df":2},
    ])
    prediction = predict_tennis_match("A", "B", matches, "Hard")
    assert prediction["pick"] in {"A", "B"}
    assert abs(prediction["a_win"] + prediction["b_win"] - 1.0) < 1e-9
    assert 0.0 <= prediction["confidence"] <= 1.0
    assert prediction["elo_a"] > 0


def test_low_sample_players_get_confidence_shrunk_toward_even():
    matches = pd.DataFrame([
        {"tourney_date":"20260101","winner_name":"NewPlayer","loser_name":"OtherNew","surface":"Hard","w_svpt":70,"w_1stWon":55,"w_2ndWon":10,"w_ace":12,"w_df":1,"l_svpt":60,"l_1stWon":25,"l_2ndWon":8,"l_ace":2,"l_df":5},
        {"tourney_date":"20260105","winner_name":"NewPlayer","loser_name":"OtherNew","surface":"Hard","w_svpt":68,"w_1stWon":50,"w_2ndWon":9,"w_ace":10,"w_df":1,"l_svpt":62,"l_1stWon":26,"l_2ndWon":9,"l_ace":3,"l_df":4},
    ])
    prediction = predict_tennis_match("NewPlayer", "OtherNew", matches, "Hard")
    assert prediction["low_sample"] is True
    assert prediction["confidence"] < 0.65
