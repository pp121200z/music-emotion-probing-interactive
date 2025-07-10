import React, { useEffect, useRef } from 'react';
import p5 from 'p5';

export default function EmotionVisualizer({ valence, arousal }) {
  const sketchRef = useRef();

  const classifyEmotion = (val, aro) => {
    if (val >= 0 && aro >= 0) return "Joy";
    if (val < 0 && aro >= 0) return "Anger";
    if (val < 0 && aro < 0) return "Sad";
    if (val >= 0 && aro < 0) return "Calm";
    return "Unknown";
  };

  const emotion = (valence !== null && arousal !== null)
    ? classifyEmotion(valence, arousal)
    : "Unknown";

  const getQuadrantCenter = (emo) => {
    switch (emo) {
      case "Joy": return { x: 300, y: 100 };
      case "Anger": return { x: 100, y: 100 };
      case "Sad": return { x: 100, y: 300 };
      case "Calm": return { x: 300, y: 300 };
      default: return { x: 200, y: 200 };
    }
  };

  useEffect(() => {
    const sketch = (p) => {
      p.setup = () => {
        p.createCanvas(400, 400);
        p.textAlign(p.CENTER, p.CENTER);
        p.textFont('Arial');
        p.textSize(14);
      };

      p.draw = () => {
        p.background(255);

        p.stroke(0);
        p.strokeWeight(2);
        p.line(200, 0, 200, 400); // vertical
        p.line(0, 200, 400, 200); // horizontal
        p.noStroke();
        p.fill(0);
        p.text("→", 392, 200);
        p.text("Valence", 350, 190);
        p.text("←", 9, 200);
        p.text("↑ Arousal", 237, 9);
        p.text("↓", 200, 390);
        const { x, y } = getQuadrantCenter(emotion);
        p.fill(150, 100, 255, 180);
        p.noStroke();
        p.ellipse(x, y, 80, 80); // emotion

        p.fill(0);
        p.textSize(20);
        p.text(emotion, x, y - 50);
      };
    };

    const p5Instance = new p5(sketch, sketchRef.current);
    return () => p5Instance.remove();
  }, [emotion]);

  return (
    <div className="flex flex-col items-center space-y-4">
      <div ref={sketchRef} />
      <div className="text-xl font-semibold text-center text-gray-800">
        Your emotion is: <span className="font-bold text-purple-600">{emotion}</span>
      </div>
    </div>
  );
}
