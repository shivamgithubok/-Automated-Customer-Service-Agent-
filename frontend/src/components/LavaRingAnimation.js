import React, { useRef, useEffect } from 'react';

const LavaRingAnimation = () => {
  // Use a ref to access the canvas DOM node
  const canvasRef = useRef(null);

  useEffect(() => {
    // Check if the canvas element exists
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Get the 2D rendering context
    const ctx = canvas.getContext('2d');
    
    // An array to hold the animated particles
    let particlesArray = [];
    
    // Animation loop reference
    let animationFrameId;
    
    // Set canvas dimensions to match the window size for a responsive background
    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    // Particle object constructor for the "lava"
    class Particle {
      constructor(x, y, vx, vy, size, color) {
        this.x = x;
        this.y = y;
        this.vx = vx;
        this.vy = vy;
        this.size = size;
        this.color = color;
      }
      
      // Method to draw a single particle
      draw() {
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2, false);
        ctx.fillStyle = this.color;
        ctx.fill();
      }
      
      // Method to update particle position
      update() {
        this.x += this.vx;
        this.y += this.vy;
        
        // Decrease size and alpha over time for a fading effect
        if (this.size > 0.1) this.size -= 0.05;
      }
    }
    
    // Function to create new "lava" particles randomly across the background
    const spawnParticles = () => {
      // Spawn a few particles at random points on the canvas
      for (let i = 0; i < 5; i++) {
        // Randomly get a spawn point within the canvas dimensions
        const spawnX = Math.random() * canvas.width;
        const spawnY = Math.random() * canvas.height;
        
        // Calculate a random velocity
        const speed = Math.random() * 2 + 1;
        const vx = (Math.random() - 0.5) * speed;
        const vy = (Math.random() - 0.5) * speed;
        
        // Choose a warm color
        const colors = ['#FF4500', '#FF8C00', '#FFD700']; // OrangeRed, DarkOrange, Gold
        const color = colors[Math.floor(Math.random() * colors.length)];
        
        const size = Math.random() * 2 + 1;
        
        particlesArray.push(new Particle(spawnX, spawnY, vx, vy, size, color));
      }
    };
    
    // The main animation loop
    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear the canvas each frame
      
      // Update and draw each particle
      for (let i = particlesArray.length - 1; i >= 0; i--) {
        particlesArray[i].update();
        particlesArray[i].draw();
        
        // Remove particles that are too small
        if (particlesArray[i].size <= 0.1) {
          particlesArray.splice(i, 1);
        }
      }
    };
    
    // Event listeners for window resize
    window.addEventListener('resize', resizeCanvas);
    
    // Set a timer to spawn new particles every 100ms
    const particleInterval = setInterval(spawnParticles, 100);

    // Initial setup on mount
    resizeCanvas();
    animate();

    // Cleanup function to stop the animation when the component unmounts
    return () => {
      cancelAnimationFrame(animationFrameId);
      clearInterval(particleInterval);
      window.removeEventListener('resize', resizeCanvas);
    };
  }, []); // Empty dependency array ensures this effect runs only once on mount

  // Render a canvas element that takes up the full space
  return <canvas ref={canvasRef} className="absolute inset-0 w-full h-full bg-transparent z-0" />;
};

export default LavaRingAnimation;
