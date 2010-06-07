
; all elements are distinct
(define test '(7 (3 () (5)) (9 () ())))



(define (entry tree) (car tree))

(define (left tree) (cadr tree))
(define (right tree) (caddr tree))

(define (contain? tree elem)
  (if (pair? tree)
      (let ((e (entry tree)))
        (cond ((equal? e elem) #t)
        ((> e elem) (contain? (left tree) elem))
        ((< e elem) (contain? (right tree) elem))
        (else #f)))
      #f))

