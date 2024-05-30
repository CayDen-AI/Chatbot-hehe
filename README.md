- cài đặt các thư viện python thì sử dụng lệnh:
```
    pip install -r setup.txt
```
<br>

- data sẽ được đưa vào file `data/fithou`
- data có định dạng file là pdf
- sau khi đưa data vào file `data/fithou` rồi thì chạy file `database.py` bằng lệnh:
```
    python database.py
```
- sau khi chạy file `database.py` sẽ xuất hiện folder `faiss`, nếu lỗi thì tạo trước rồi chạy lại lệnh
- sau khi đã có database rồi thì chạy file `main.py` bằng lệnh:
```
    python main.py
```
- rồi ấn vào cổng `http://127.0.0.1:5000` để chạy trên web
<br>

**Hình minh họa**
<img src='image/Screenshot 2024-05-30 131638.png'>
<img src='image/Screenshot 2024-05-30 135249.png'>