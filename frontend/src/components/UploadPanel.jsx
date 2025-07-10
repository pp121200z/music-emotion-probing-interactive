import React, { useState } from 'react';

export default function UploadPanel({ onResult }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch("http://localhost:5000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (!data.valence || !data.arousal) {
        alert("Invalid");
        return;
      }

      onResult(data);
    } catch (err) {
      console.error("Request Failed:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center h-full space-y-8">
      {/* Upload Button*/}
      <label className="w-full text-center">
        <input
          type="file"
          accept=".csv"
          onChange={e => setFile(e.target.files[0])}
          className="hidden"
          id="file-upload"
        />
        <span className="inline-block w-full bg-white border border-gray-400 hover:bg-gray-50 text-xl text-gray-800 py-5 px-10 rounded cursor-pointer font-semibold">
          Choose File
        </span>
      </label>

      {/* Detect Button*/}
      <button
        className="w-full bg-blue-600 hover:bg-blue-700 text-white text-xl font-bold py-5 px-10 rounded disabled:bg-gray-400 transition"
        onClick={handleUpload}
        disabled={!file || loading}
      >
        {loading ? 'Detecting...' : 'Detect Emotion'}
      </button>
    </div>
  );
}

