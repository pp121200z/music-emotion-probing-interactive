import React, { useState } from 'react';
import UploadPanel from './components/UploadPanel';
import EmotionVisualizer from './components/EmotionVisualizer';

export default function App() {
  const [result, setResult] = useState(null);

  const handleResult = (data) => {
    setResult(data);
  };

  return (
    <div className="min-h-screen bg-gray-100 flex justify-center items-center px-4">
      <div className="w-[960px] bg-white rounded-xl shadow-lg p-10 flex space-x-10">

        {/* Left card - Upload Area */}
        <div className="w-[300px] bg-white rounded-lg border border-gray-300 p-6 shadow-sm flex flex-col justify-between">
          <h2 className="text-2xl font-bold text-gray-800 mb-6 text-center">Emotion Detection</h2>
          <UploadPanel onResult={handleResult} />
          <p className="text-sm text-gray-500 mt-6 text-center">
            Upload physiological data and detect emotional states using our ML model.
          </p>
        </div>

        {/* Right card - Visualizer */}
        <div className="w-[580px] bg-white rounded-lg border border-gray-300 p-6 shadow-sm flex flex-col justify-between">
          <h2 className="text-2xl font-bold text-gray-800 mb-4 text-center">Emotion Visualization</h2>
          <EmotionVisualizer
            valence={result?.valence ?? null}
            arousal={result?.arousal ?? null}
          />
        </div>
      </div>
    </div>
  );
}

