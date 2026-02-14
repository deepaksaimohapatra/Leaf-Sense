/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f0f9f1',
          100: '#dcf1df',
          200: '#bbe3c3',
          300: '#8fce9d',
          400: '#5fb172',
          500: '#3d9453',
          600: '#2e7841',
          700: '#266036',
          800: '#214d2e',
          900: '#1c3f28',
          950: '#0f2316',
        },
      }
    },
  },
  plugins: [],
}
