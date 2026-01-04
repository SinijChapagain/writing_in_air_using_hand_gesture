# Gesture-Based Note-Taking System
The development of intuitive human-computer interaction methods has become increasingly important as digital technologies permeate everyday life. Traditional input methods such as keyboards and touchscreens, while effective, present limitations in accessibility, hygiene, and ergonomic contexts. This project addresses these challenges through the implementation of a contactless gesture recognition system that enables users to create and modify digital notes using natural hand movements.

A contactless interface for drawing and erasing digital notes using hand gestures: no keyboard, no mouse, just natural hand movements.

[Demo: Pinch to draw, open palm to erase] 
*(Example: Pinch: draw blue line | Open palm: erase with red circle)*

### Prerequisites
- Python 3.9+
- macOS (for AVFoundation camera support)
- `pip` package manager

### Project Structure

gesture_note_taking/
- demo_rt.py           (Real-time demo)
- model.py             (GestureNet architecture)
- train_torch.py       (Training script)
- extract_sequences.py (Data preprocessing)
- record.py          (Record your own gestures)
- X_seq.npy          (Preprocessed training data)
- y_seq.npy          (Training labels)
- label_encoder.pkl (Label mapping (erase=0, note=1))
- gesture_model.pth (Trained model weights)
- requirements.txt (Dependencies)

### Technical Details
- Model: 1D CNN (GestureNet) for temporal landmark sequences
- Input: 16-frame sequences of 63 hand landmark coordinates
- Output: Binary classification (note/erase)
- Inference: 4.2ms per prediction (238 FPS theoretical)
- Hardware: Tested on Apple M1 MacBook Pro

### Installation
```bash
# Clone the repository
git clone https://github.com/SinijChapagain/writing_in_air_using_hand_gesture.git
cd writing_in_air_using_hand_gesture

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run the demo
python demo_rt.py
- Pinch (thumb + index touching) → draw
- Open palm (back of hand facing camera) → erase
- Press q to quit


