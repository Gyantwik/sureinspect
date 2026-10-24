from eval.download_visa_pcb2 import destination_for, official_test_rows


def test_official_test_rows_selects_only_pcb2_test():
    payload = (
        b"object,split,label,image,mask\n"
        b"pcb2,train,normal,pcb2/Data/Images/Normal/0000.JPG,\n"
        b"pcb2,test,normal,pcb2/Data/Images/Normal/0999.JPG,\n"
        b"pcb2,test,anomaly,pcb2/Data/Images/Anomaly/000.JPG,mask.png\n"
        b"pcb1,test,normal,pcb1/Data/Images/Normal/0999.JPG,\n"
    )
    rows = official_test_rows(payload)
    assert [row["label"] for row in rows] == ["normal", "anomaly"]
    assert destination_for(rows[0]).as_posix().endswith("data/visa/pcb2/test/good/0999.JPG")
    assert destination_for(rows[1]).as_posix().endswith("data/visa/pcb2/test/bad/000.JPG")
