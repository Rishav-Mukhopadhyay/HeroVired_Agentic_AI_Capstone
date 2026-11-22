from mediscribe.data_sources import clean_aci_text, parse_dialogue


def test_clean_aci_text_fixes_tokenization():
    assert clean_aci_text("i , i do n't know . it 's fine") == "i, i don't know. it's fine"


def test_parse_dialogue_maps_speakers():
    turns = parse_dialogue("[doctor] hi , brian .\n[patient] hello .\n[patient_guest] i'm his wife .")
    assert [t["speaker"] for t in turns] == ["doctor", "patient", "other"]
    assert turns[0]["text"] == "hi, brian."
