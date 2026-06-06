# --- Build Stage ---
FROM golang:1.26-alpine AS builder

# Install build tools
RUN apk add --no-cache git build-base

WORKDIR /app

# Copy dependency definition files
COPY go.mod go.sum ./
RUN go mod download

# Copy the rest of the source code
COPY . .

# Build the Go application binary with optimizations (restricted resources to prevent OOM on 1GB RAM VM)
RUN GOMAXPROCS=1 GOMEMLIMIT=500MiB CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags="-w -s" -o main .


# --- Run Stage ---
FROM alpine:3.19

# Add certificates for secure HTTPS external API requests (e.g., OpenRouter) and timezone data
RUN apk --no-cache add ca-certificates tzdata

WORKDIR /app

# Copy the compiled binary from the builder stage
COPY --from=builder /app/main .

# Copy config, prompts, and docs directories needed at runtime
COPY --from=builder /app/config ./config
COPY --from=builder /app/prompts ./prompts
COPY --from=builder /app/docs ./docs

# Set runtime port default (can be overridden via environment variable)
ENV PORT=8080

# Expose API port
EXPOSE 8080

# Run the Go API gateway binary
CMD ["./main"]
