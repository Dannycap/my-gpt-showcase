// netlify/functions/chat.js

import fetch from 'node-fetch';

export const handler = async (event) => {
  try {
    const { ticker } = JSON.parse(event.body);

    const apiRes = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`
      },
      body: JSON.stringify({
        model: 'gpt-4',
        messages: [
          { role: 'system', content: 'You are a seasoned financial analyst.' },
          { role: 'user', content: `Analyze ${ticker}: recent performance, P/E, and risks.` }
        ],
        temperature: 0.7,
        max_tokens: 300
      })
    });

    const data = await apiRes.json();
    return { statusCode: apiRes.status, body: JSON.stringify(data) };

  } catch (err) {
    return { statusCode: 500, body: JSON.stringify({ error: err.message }) };
  }
};
