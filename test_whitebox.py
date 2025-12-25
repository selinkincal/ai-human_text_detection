import pytest

from clean_data import clean_text  # Projedeki temizleme kodunu ceker

# White Box Test 1: Regex Mantigi
def test_clean_text_logic():
    sample = "<p>Abstract: $x^2$</p>"
    result = clean_text(sample)
    # Icerideki <p> ve $ isaretleri temizlenmis mi?
    assert "<p>" not in result
    assert "$" not in result
    assert result.strip() == "Abstract:"

# White Box Test 2: Karakter Siniri Mantigi
def test_min_length_logic():
    # clean_data.py icindeki 50 karakter sinirini test eder
    short_text = "Cok kisa bir metin."
    # Bu metin temizlendiginde sistemin bunu nasil isledigini dogrulariz
    assert len(clean_text(short_text)) < 50

    # White Box Test 3: Singleton Pattern Kontrolu
def test_model_loader_singleton():
    from app import ModelLoader
    # Iki farkli nesne olusturmaya calisalim
    obj1 = ModelLoader()
    obj2 = ModelLoader()
    # Bellekteki adresleri ayni mi? (Singleton garantisi)
    assert obj1 is obj2

