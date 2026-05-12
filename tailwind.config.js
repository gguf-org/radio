/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      boxShadow: {
        "active-glow": "0 0 0 1px rgba(103,232,249,0.15), 0 14px 36px rgba(8,51,68,0.28)",
        "panel-soft": "0 24px 70px rgba(8,51,68,0.32), 0 2px 10px rgba(0,0,0,0.3)",
      },
    },
  },
  plugins: [],
};
