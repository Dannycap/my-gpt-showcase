import { useState } from 'react';
import axios from 'axios';
import { Scatter } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  PointElement,
  LinearScale,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(PointElement, LinearScale, Tooltip, Legend);

export default function Home() {
  const [testSize, setTestSize] = useState(0.33);
  const [frontierSize, setFrontierSize] = useState(30);
  const [data, setData] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const resp = await axios.post(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/efficient-frontier`, {
      test_size: testSize,
      efficient_frontier_size: frontierSize,
    });
    setData(resp.data);
  };

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Efficient Frontier Explorer</h1>
      <form onSubmit={handleSubmit} className="space-y-2">
        <div>
          <label className="block">Test Size</label>
          <input
            type="number"
            step="0.01"
            value={testSize}
            onChange={(e) => setTestSize(parseFloat(e.target.value))}
            className="border p-1"
          />
        </div>
        <div>
          <label className="block">Frontier Size</label>
          <input
            type="number"
            value={frontierSize}
            onChange={(e) => setFrontierSize(parseInt(e.target.value))}
            className="border p-1"
          />
        </div>
        <button className="bg-blue-500 text-white px-4 py-2">Run</button>
      </form>
      {data && (
        <div className="mt-4">
          <h2 className="font-semibold">Summary</h2>
          <table className="min-w-full text-sm border">
            <thead>
              <tr>
                {Object.keys(data.summary[0]).map((col) => (
                  <th key={col} className="border px-2">
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.summary.map((row, idx) => (
                <tr key={idx}>
                  {Object.values(row).map((val, i) => (
                    <td key={i} className="border px-2">
                      {typeof val === 'number' ? val.toFixed(4) : val}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <div className="h-96 mt-4">
            <Scatter
              data={{
                datasets: [
                  {
                    label: 'Portfolios',
                    data: data.plot.risk.map((x, i) => ({ x, y: data.plot.return[i] })),
                    backgroundColor: 'rgba(37,99,235,0.5)',
                  },
                ],
              }}
              options={{
                scales: {
                  x: { title: { display: true, text: 'Risk' } },
                  y: { title: { display: true, text: 'Return' } },
                },
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
