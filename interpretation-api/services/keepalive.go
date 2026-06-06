package services

import (
	"log"
	"net/http"
	"strings"
	"time"
)

// StartDailyKeepAlive starts a background goroutine that pings the target URL every 24 hours.
func StartDailyKeepAlive(targetURL string) {
	// Bypass keep-alive if targetURL is local (development) or empty
	if targetURL == "" || 
		strings.Contains(targetURL, "localhost") || 
		strings.Contains(targetURL, "127.0.0.1") || 
		strings.Contains(targetURL, "::1") {
		log.Printf("[*] Keep-alive scheduler bypassed: target URL %q is local or empty.", targetURL)
		return
	}

	go func() {
		log.Printf("[*] Hugging Face keep-alive scheduler started for URL: %s", targetURL)
		
		// Run initial ping on startup to verify and awaken the container
		pingHF(targetURL)

		// Create a ticker to run every 24 hours
		ticker := time.NewTicker(24 * time.Hour)
		defer ticker.Stop()

		for range ticker.C {
			pingHF(targetURL)
		}
	}()
}

func pingHF(url string) {
	log.Printf("[*] Sending keep-alive request to: %s", url)
	
	// Create client with timeout to ensure keep-alive ping doesn't hang indefinitely
	client := &http.Client{
		Timeout: 30 * time.Second,
	}

	resp, err := client.Get(url)
	if err != nil {
		log.Printf("[!] Keep-alive ping failed: %v", err)
		return
	}
	defer resp.Body.Close()

	log.Printf("[+] Keep-alive ping completed. Response Status: %d (%s)", resp.StatusCode, resp.Status)
}
