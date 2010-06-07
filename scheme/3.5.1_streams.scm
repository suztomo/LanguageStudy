(define (prime? x)
  (let ((limit (sqrt x)))
    (define iter (lambda (i)
                   (cond
                    ((> i limit) #t)
                    ((= 0 (modulo x i))
                     #f)
                    (else (iter (+ i 1))))))
    (iter 2)))

(define (stream-enumerate-interval low high)
  (if (> low high)
      the-empty-stream
      (cons-stream low
                   (delay (stream-enumerate-interval (+ low 1) high)))))

(define the-empty-stream
  ())

(define (stream-null? s)
  (null? s))

(define (stream-map proc s)
  (if (stream-null? s)
      the-empty-stream
      (cons-stream (proc (stream-car s))
                   (stream-map proc (stream-cdr s)))))

(define (stream-for-each proc s)
  (if (stream-null? s)
      'done
      (begin (proc (stream-car s))
             (stream-for-each proc (stream-cdr s)))))


(define (display-stream s)
  (stream-for-each display-line s))

(define (display-line x)
  (newline)
  (display x))


(define (cons-stream a b)
  (cons a b))

(define (stream-car stream)
  (car stream))

(define (stream-filter filter stream)
  (if (stream-null? stream)
      the-empty-stream
      (if (filter (stream-car stream))
          (cons-stream (stream-car stream)
                       (delay (stream-filter filter (stream-cdr stream))))
          (stream-filter filter (stream-cdr stream)))))

(define (stream-cdr stream)
  (force (cdr stream)))

(stream-car
 (stream-cdr
  (stream-filter prime?
                 (stream-enumerate-interval 1000 1000000))))

(car (cdr (filter prime?
                  (enumerate-interval 10000 1000000))))


(define tako
  (cons-stream 1
             (delay (cons-stream 3
                          (delay (cons-stream 5 the-empty-stream))))))
(stream-car (stream-cdr tako))

(define (func count)
  (if (= count 0)
      0
      (begin
        (display "Hello, world")
        (func (- count 1)))
      ))
