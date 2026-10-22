## 📝 Deskripsi

Jelaskan perubahan Anda secara singkat dan jelas.

## 🎯 Jenis Perubahan

Pilih yang relevan:

- [ ] 🐛 Bug fix (perubahan non-breaking yang memperbaiki issue)
- [ ] ✨ New feature (perubahan non-breaking yang menambahkan functionality)
- [ ] 💥 Breaking change (fix atau feature yang menyebabkan existing functionality tidak bekerja seperti sebelumnya)
- [ ] 📝 Documentation update
- [ ] ♻️ Code refactoring
- [ ] ✅ Test improvements
- [ ] 🎨 Style/UI improvements
- [ ] ⚡️ Performance improvements

## 🔗 Related Issues

Closes #(issue number)

## 📸 Screenshots / Demo (jika applicable)

Tambahkan screenshots atau GIF yang menunjukkan perubahan Anda.

## 🧪 Testing

Jelaskan testing yang sudah Anda lakukan:

- [ ] Unit tests pass locally
- [ ] Integration tests pass locally
- [ ] Manual testing completed

**Test commands:**
```bash
python -m pytest tests/ -v
```

**Test results:**
```
# Paste test output atau summary di sini
```

## ✅ Checklist

Sebelum submit PR, pastikan:

### Code Quality
- [ ] Kode mengikuti style guidelines project
- [ ] Self-review sudah dilakukan
- [ ] Komentar ditambahkan di area yang sulit dipahami
- [ ] Tidak ada console.log/print statements yang tertinggal (kecuali memang diperlukan)

### Documentation
- [ ] Docstrings ditambahkan/diupdate untuk public APIs
- [ ] README.md diupdate jika ada perubahan user-facing
- [ ] CHANGELOG.md diupdate (jika ada)

### Testing
- [ ] Tests ditambahkan untuk fitur baru / bug fix
- [ ] Semua tests pass (local dan CI)
- [ ] Test coverage tidak menurun
- [ ] Edge cases sudah dipertimbangkan

### Compatibility
- [ ] Backward compatibility dijaga (atau dijelaskan breaking changes)
- [ ] Tidak ada conflict dengan main branch
- [ ] Dependencies baru (jika ada) sudah didokumentasikan

### Additional (jika applicable)
- [ ] Algoritma baru mengikuti struktur folder standar
- [ ] README algorithm dalam Bahasa Indonesia
- [ ] Exercise notebook ditambahkan
- [ ] Dataset sample disediakan (jika diperlukan)

## 📊 Performance Impact (jika applicable)

Jelaskan dampak perubahan terhadap performance:

- Memory usage: [increase/decrease/same]
- Execution time: [faster/slower/same]
- Benchmarks:
  ```
  # Paste benchmark results jika ada
  ```

## 🔄 Migration Guide (jika breaking change)

Jelaskan bagaimana users harus migrate kode mereka:

```python
# Before
old_way()

# After
new_way()
```

## 💭 Additional Context

Tambahkan konteks lain yang perlu diketahui reviewer:
- Design decisions
- Trade-offs yang dipertimbangkan
- Future improvements yang planned
- Known limitations

## 👀 Reviewer Notes

Area yang perlu perhatian khusus dari reviewer:
- [ ] Area 1: ...
- [ ] Area 2: ...

---

**Terima kasih atas kontribusi Anda! 🙏**
