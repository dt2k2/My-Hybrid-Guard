# URL Test Data Sources

Danh sach nguon dataset/feed de tao test case co nhan va test case stream:

## 1) Dataset co nhan (offline benchmark)

- Kaggle Malicious URLs dataset (651k URL):
  - https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset
  - Co cot `url` va `type` (benign/phishing/malware/defacement)

- Dataset dang co san trong project:
  - Dataset.csv
  - Co cot label 0/1 va cac feature URL

## 2) Feed phishing/malware de bo sung test case

- PhishTank developer feed:
  - https://www.phishtank.com/developer_info.php
  - Format: CSV/JSON/XML, co danh sach URL phishing da verify
  - Vi du feed: http://data.phishtank.com/data/online-valid.json

- OpenPhish community feed (free):
  - https://openphish.com/phishing_feeds.html
  - Free txt feed: https://raw.githubusercontent.com/openphish/public_feed/refs/heads/main/feed.txt

- URLhaus API / dump:
  - https://urlhaus.abuse.ch/api/
  - Co plain-text URL list, CSV, JSON (yeu cau auth-key)

## 3) Luu y khi dung feed

- Feed thuong la du lieu mot lop (chu yeu malicious), can bo sung benign list de tinh FP/FPR.
- Neu dung feed online de benchmark hybrid mode, nen gioi han sample-size va bat cache de tranh rate limit.
- Nen tao tap test can bang (benign vs malicious) de so sanh cac model cong bang hon.
