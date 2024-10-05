import tkinter as tk
from tkinter import messagebox
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def dudoan():
    try:
        so_gio_on_tap = float(so_gio_on_tap_entry.get())
        diem_trung_binh = float(diem_trung_binh_entry.get())
        so_lan_nghi_hoc = int(so_lan_nghi_hoc_entry.get())
        
        # Áp dụng các quy tắc ưu tiên
        if so_gio_on_tap > 3:
            messagebox.showinfo("Dự đoán", "Học sinh sẽ đậu lớp vì đã học nhiều.")
        elif so_lan_nghi_hoc > 5:
            messagebox.showinfo("Dự đoán", "Học sinh sẽ không đậu lớp vì nghỉ học quá nhiều.")
        elif diem_trung_binh >= 9.5:
            messagebox.showinfo("Dự đoán", "Học sinh sẽ đậu lớp vì điểm cao.")
        else:
            # Nếu không thỏa mãn các điều kiện ưu tiên, sử dụng mô hình để dự đoán
            du_lieu_nhap_chuan_hoa = scaler.transform([[so_gio_on_tap, diem_trung_binh, so_lan_nghi_hoc]])
            du_doan = clf.predict(du_lieu_nhap_chuan_hoa)
            
            if du_doan[0] == 1:
                messagebox.showinfo("Dự đoán", "Học sinh sẽ đậu lớp.")
            else:
                messagebox.showinfo("Dự đoán", "Học sinh sẽ không đậu lớp.")
    
    except ValueError:
        messagebox.showerror("Lỗi", "Vui lòng nhập đúng định dạng cho các trường.")

def giai_thich_du_doan(so_gio_on_tap, diem_trung_binh, so_lan_nghi_hoc):
    feature_names = ["Số giờ ôn tập", "Điểm trung bình", "Số lần nghỉ học"]
    # Dự đoán nhãn
    du_lieu_nhap_chuan_hoa = scaler.transform([[so_gio_on_tap, diem_trung_binh, so_lan_nghi_hoc]])
    du_doan = clf.predict(du_lieu_nhap_chuan_hoa)
    
    # Giải thích quyết định
    node_indicator = clf.decision_path(du_lieu_nhap_chuan_hoa)
    leaf_id = clf.apply(du_lieu_nhap_chuan_hoa)
    
    # Lấy đường dẫn từ nút gốc đến nút lá
    node_index = node_indicator.indices
    print(f"Dự đoán: {'Đậu' if du_doan[0] == 1 else 'Không đậu'}")
    print("Đường dẫn từ nút gốc đến nút lá:")
    for node_id in node_index:
        if leaf_id[0] == node_id:
            continue
        if (du_lieu_nhap_chuan_hoa[0, clf.tree_.feature[node_id]] <= clf.tree_.threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"
        print(f"{' ' * clf.tree_.max_depth * 2} - {feature_names[clf.tree_.feature[node_id]]} {threshold_sign} {clf.tree_.threshold[node_id]}")


def hien_thi_cay_quyet_dinh():
    plt.figure(figsize=(5,5))
    plot_tree(clf, filled=True, rounded=True, feature_names=["Số giờ ôn tập", "Điểm trung bình", "Số lần nghỉ học"], class_names=["Không đậu", "Đậu"])
    plt.show()

# Ví dụ: Số giờ ôn tập, Điểm trung bình, Số lần nghỉ học, 
data = [
    [2, 8, 0],   # Học sinh 1
    [1, 7, 2],   # Học sinh 2
    [3, 8, 1],   # Học sinh 3
    [0, 6, 0],   # Học sinh 4
    [4, 9, 0],   # Học sinh 5
    [2, 7, 1],   # Học sinh 6
    [3, 9, 2],   # Học sinh 7
    [2, 8, 1],   # Học sinh 8
    [1, 7, 0],   # Học sinh 9
    [2, 8, 2],   # Học sinh 10
    [3, 9, 1],   # Học sinh 11
    [0, 6, 1],   # Học sinh 12
    [4, 9, 2],   # Học sinh 13
    [2, 7, 0],   # Học sinh 14
    [3, 8, 0],   # Học sinh 15
    [2, 8, 1],   # Học sinh 16
    [1, 7, 1],   # Học sinh 17
    [3, 9, 0],   # Học sinh 18
    [2, 8, 2],   # Học sinh 19
]

# Nhãn cho mỗi mẫu dữ liệu: 1 nếu đậu, 0 nếu trượt
nhan = [1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]


# Khởi tạo giao diện
giao_dien = tk.Tk()
giao_dien.title("Dự đoán việc đậu lớp")
giao_dien.geometry("550x500")  # Đặt kích thước của cửa sổ giao diện

# Khởi tạo các điều khiển giao diện...
so_gio_on_tap_label = tk.Label(giao_dien, text="Số giờ ôn tập:", font=("Arial", 14))
so_gio_on_tap_label.grid(row=0, column=0, pady=5)
so_gio_on_tap_entry = tk.Entry(giao_dien, font=("Arial", 14))
so_gio_on_tap_entry.grid(row=0, column=1, pady=5)

diem_trung_binh_label = tk.Label(giao_dien, text="Điểm trung bình:", font=("Arial", 14))
diem_trung_binh_label.grid(row=1, column=0, pady=5)
diem_trung_binh_entry = tk.Entry(giao_dien, font=("Arial", 14))
diem_trung_binh_entry.grid(row=1, column=1, pady=5)

so_lan_nghi_hoc_label = tk.Label(giao_dien, text="Số lần nghỉ học:", font=("Arial", 14))
so_lan_nghi_hoc_label.grid(row=2, column=0, pady=5)
so_lan_nghi_hoc_entry = tk.Entry(giao_dien, font=("Arial", 14))
so_lan_nghi_hoc_entry.grid(row=2, column=1, pady=5)

du_doan_button = tk.Button(giao_dien, text="Dự đoán", command=dudoan, font=("Arial", 14))
du_doan_button.grid(row=4, columnspan=2, pady=10)

# Button để hiển thị cây quyết định
hien_thi_cay_quyet_dinh_button = tk.Button(giao_dien, text="Hiển thị cây quyết định", command=hien_thi_cay_quyet_dinh, font=("Arial", 14))
hien_thi_cay_quyet_dinh_button.grid(row=5, columnspan=2, pady=10)

# Load mô hình và chuẩn hóa dữ liệu
scaler = preprocessing.StandardScaler().fit(data)
du_lieu_chuan_hoa = scaler.transform(data)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(du_lieu_chuan_hoa, nhan)

# Chạy giao diện
giao_dien.mainloop()
