# Data Directory

This directory contains the datasets used for training and testing.

## 📊 Active Datasets

### `Aalto_train_IoTDevID (1).csv` (17.4 MB)
- **Purpose**: Training data
- **Samples**: ~22,000 network traffic flows
- **Classes**: 27 IoT device types
- **Features**: 96 network traffic features
- **Status**: ✅ **ACTIVE - Used for training**

**Device Classes Include:**
- HueSwitch, HueBridge (Smart lighting)
- D-Link devices (Cameras, Sensors, Hubs)
- Ednet devices (Cameras, Gateway)
- TP-Link plugs (Smart plugs)
- WeMo devices (Smart switches)
- And 17+ more device types

---

### `Aalto_test_IoTDevID (1).csv` (5.5 MB)
- **Purpose**: Validation/Test data
- **Samples**: 19,932 network traffic flows
- **Usage**: Used as validation set during training
- **Status**: ✅ **ACTIVE - Used for validation**

**Note**: Despite the name "test", this file is used as the validation set during training (specified in `train_hist_gb.py` as `DEFAULT_VALID_PATH`).

---

### `veto_average_results (1).csv`
- **Purpose**: Feature importance rankings
- **Content**: 96 features ranked by importance votes
- **Usage**: 
  - Training scripts select top-K features (default: 50)
  - API loads this to know which features to use
- **Status**: ✅ **ACTIVE - Required for all models**

**Top Features Include:**
- TCP/UDP ports
- IP protocol fields
- Packet sizes
- Timing features
- Entropy measures

---

## 🔍 Data Format

Each CSV row represents one network flow with:
- **Network Features** (96 columns): TCP_sport, TCP_dport, IP_proto, etc.
- **Label** (1 column): Device type name (e.g., "HueSwitch")
- **Metadata** (3 columns): MAC address, Protocol type

---

## 📝 Feature Selection

Models use **top 48-50 features** from the veto rankings, not all 96:

```python
# Top 10 most important features:
1. TCP_sport      - Source port
2. TCP_dport      - Destination port  
3. UDP_sport      - UDP source port
4. IP_proto       - IP protocol number
5. UDP_len        - UDP packet length
6. TCP_seq        - TCP sequence number
7. IP_id          - IP identification
8. TCP_window     - TCP window size
9. UDP_chksum     - UDP checksum
10. dport_class   - Destination port class
```

---

## ⚠️ Important Notes

1. **NaN Values**: Some features contain NaN (e.g., TCP fields when packet is UDP)
   - Training fills NaN with 0.0
   - API must do the same!

2. **Mixed Types**: Column 35 has mixed types (warned during training)
   - Not an issue for our selected features

3. **File Size**: Training data is 17MB - keep in repo but consider git-lfs for larger datasets

4. **No True Test Set**: The "test" CSV is used for validation. For true testing, use a separate holdout set or cross-validation.

---

## 🧪 Quick Data Check

```python
import pandas as pd

# Load and inspect
train_df = pd.read_csv('data/Aalto_train_IoTDevID (1).csv')
print(f"Training samples: {len(train_df)}")
print(f"Device types: {train_df['Label'].nunique()}")
print(f"\nDevice distribution:")
print(train_df['Label'].value_counts())
```

---

## 📈 Data Statistics

- **Training**: ~22,000 samples
- **Validation**: ~20,000 samples  
- **Total**: ~42,000 network flows
- **Classes**: 27 device types (balanced for some, imbalanced for others)
- **Features**: 96 raw features → 48 selected
- **Time Period**: Captured over several days/weeks
