/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}",
  ],
  theme: {
    extend: {
      colors: {
        saffron:   { DEFAULT: '#FF9933', dark: '#CC7A00' },
        navy:      { DEFAULT: '#1B2A4A', light: '#2D4270' },
        parchment: { DEFAULT: '#F5F0E8', dark: '#E8E0CC' },
      },
      fontFamily: {
        sans:     ['Inter', 'system-ui', 'sans-serif'],
        serif:    ['Merriweather', 'Georgia', 'serif'],
        samarkan: ['Samarkan', 'serif'],
      },
    },
  },
  plugins: [require('@tailwindcss/typography')],
}
